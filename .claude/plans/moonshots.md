# moonshots — breaking the 4218 ceiling

premise: 4218 is Philip Turner's measured simdgroup_matrix throughput.
but the ALU peak is 5308 GFLOPS. the gap (1090 GFLOPS) is real silicon.

## ideas

### 1. software-pipelined MMA (HIGH PRIORITY)
current inner loop:
  load at0, at1 from tA
  load bt0, bt1 from tB
  MMA × 4

problem: loads and MMA are serialized. GPU out-of-order exec helps
but explicit interleaving may do better:
  load at0
  load bt0
  MMA acc[0][0] (starts while bt1 loading)
  load bt1
  MMA acc[0][1] (starts while at1 loading)
  load at1
  MMA acc[1][0]
  MMA acc[1][1]
  → keeps MMA unit fed while loads in flight

### 2. async threadgroup copy (HIGH PRIORITY)
Metal has threadgroup_async_copy() — uses DMA engine, NOT ALU.
our current loads use ALU threads to copy device→threadgroup.
async copy frees ALU for MMA while DMA loads next tile.
this is the REAL double buffering — not manual barriers.

### 3. stream-K for non-square matmul (HIGH for Q4)
512×4096 × 4096×11008: tiles_m=8, tiles_n=172 = 1376 tiles.
with 16 GPU cores, load imbalance: 1376/16 = 86 tiles/core.
stream-K: split K dimension across threadgroups.
each TG does partial K reduction, atomically accumulates.
better load balance + more parallelism.

### 4. 2×MMA interleaving (MOONSHOT)
each simdgroup has 32 threads. MMA uses all 32.
but what if we process TWO independent output tiles per sg?
accumulate acc_A and acc_B alternately:
  MMA acc_A[0][0] (latency: N cycles)
  MMA acc_B[0][0] (starts immediately, different accumulators)
  → 2× MMA throughput if hardware supports dual-issue

### 5. exploit SLC priming
2048×2048: A+B = 16MB, SLC = 24MB. fits!
prime SLC with a dummy read pass before matmul.
all subsequent K-iterations read from SLC at ~400 GB/s (estimate)
instead of DRAM at 200 GB/s.

### 6. Metal Performance Primitives (mpp::tensor_ops)
llama.cpp has GGML_METAL_HAS_TENSOR path using Apple's internal
mpp::tensor_ops::matmul2d. this is Apple's own optimized matmul.
test what it achieves — if it beats us, reverse-engineer why.

### 7. per-core tile affinity
current: random tile→core assignment by GPU scheduler.
what if adjacent tiles go to same core → shared A/B rows in L2?
use grid dispatch order to encourage spatial locality.

### 8. 3×3 accumulator grid per simdgroup
currently: 2×2 of 8×8 = 16×16 per sg, 4 accumulators.
try: 3×2 = 24×16 per sg, 6 accumulators. more work per sg.
BM=48, BN=64? or BM=64, BN=48? non-power-of-2 tiles.

### 9. mixed-precision MMA pipeline
accumulate in float (simdgroup_float8x8), load in half.
we tested this — was slightly slower due to register pressure.
but: what if we mix — first pass accumulate in half (fast),
then do correction pass in float? two-pass with higher throughput?

### 10. dispatch multiple matmuls simultaneously
apple GPU can run multiple command buffers concurrently.
split 2048×2048 into 4 independent 1024×2048 × 2048×1024 matmuls.
each runs in parallel, then concat results.
uses all 16 cores simultaneously with independent work.
