# rmetal performance plan

target: fastest Metal GPU compute on 1B+ Apple devices.

## current numbers (M1 Pro 16-core)

```
hardware ceiling (simdgroup_matrix):  4,218 GFLOPS
fp16 matmul (our peak cold):          3,863 GFLOPS  (91.6% ceiling)
fp16 matmul (sustained):              3,500 GFLOPS  (83%)
Q4 matmul (prefill, 512×4096×11008):  3,189 GFLOPS  (row-major B)
Q4 matvec batch=1 (decode):           247 GFLOPS, 69 GB/s, ~29 tok/s
Q4 matvec batch=8 (decode):           718 GFLOPS, ~83 tok/s  ← 2.4× llama.cpp
Q4 matvec batch=16 (decode):          814 GFLOPS, ~94 tok/s
llama.cpp Q4_K decode:                ~30-35 tok/s on same hardware
```

## what worked (keep doing)

- simdgroup_multiply_accumulate — hardware 8x8 MMA
- half4 vectorized cooperative loads
- half accumulation (simdgroup_half8x8) — less register pressure
- no bounds checking for aligned dims — +30% single biggest win
- hand-unrolled MMA — +4% from eliminating loop overhead
- bit-shift addressing — replace div/mod with shift/mask
- BK=32 BM=64 BN=64 — universal sweet spot
- row-major B layout for matvec — coalesced reads
- batched decode (dequant-once-dot-many) — batch=8 sweet spot, 2.4× llama.cpp
- register-cached weights (32 floats per Q4 block, reused across batch rows)

## what failed (don't repeat)

- half8 — does not exist in Metal (reserved type)
- double buffering — overhead > gain on Apple GPU
- 128×128 tiles — register pressure kills occupancy
- 4×4 per-sg tiles (16 accumulators) — blows register file
- LUT dequant — constant memory bank conflicts at 256 threads
- cooperative X in threadgroup memory — barriers cost more than redundant X reads
- 1024 threads per threadgroup for matvec — too few threadgroups, cores starve
- 32-byte padded Q4 blocks — 44% bandwidth waste

## open directions

### A. matvec decode (biggest impact on tok/s)

current: 247 GFLOPS, 69 GB/s (34% of 200 GB/s).
finding: compute-bound by nibble extraction, NOT memory-bound.
18 bytes per block / 64 FLOPs = 0.28 B/FLOP → theoretical max 714 GFLOPS.

ideas to try:
1. **multiple output rows per thread** — amortize dequant across 2-4 columns.
   each thread reads same B block, dots with 2-4 different X rows (batch>1 decode).
2. **SIMD cooperative dequant** — one simdgroup dequants one block (32 threads
   each extract 1 nibble), broadcast via simd_shuffle. eliminates redundant extraction.
3. **pre-dequantized cache** — for hot layers, keep fp16 weights in SLC-sized buffers.
   24 MB SLC fits ~1.5 layers of 7B Q4. trade memory for zero dequant cost.
4. **bitfield extract intrinsic** — Metal has extract_bits(). may be faster than mask+shift.
5. **Q4_K format** — 256-element super-blocks with 6-bit scales. more complex dequant
   but standard for llama.cpp. needed for compatibility.

### B. matmul prefill (already winning, push further)

current: 3,393 GFLOPS Q4. ceiling: ~4,218.
gap: dequant in load phase costs ~20% vs pure fp16.

ideas:
1. **dequant in registers, not threadgroup** — avoid writing 32 halfs to tB,
   instead dequant directly into simdgroup_load source.
2. **wider K-block for Q4** — BK=64 (2 Q4 blocks per iteration), amortize barrier cost.
3. **stream-K decomposition** — split K dimension across threadgroups for better
   load balance on non-square matrices. important for ffn (4096×11008).

### C. fp16 matmul (last 8% to ceiling)

current: 3,863 cold / 3,500 sustained. ceiling: 4,218.
remaining gap is thermal throttle + occupancy.

ideas:
1. **Xcode Metal GPU profiler** — identify exact stall cycles (memory, ALU, barrier).
2. **persistent kernel** — one dispatch, threadgroups loop over tiles. eliminates
   command buffer overhead and launch latency.
3. **tune for M1 Pro 16 cores** — threadgroup count should be multiple of 16.
   current 64×64 tiles: 1024 threadgroups for 2048. try grid that maps to 16/32/48.

### D. inference pipeline (practical product)

once kernels are solid:
1. **quantized layer struct** — Q4 weights + fp16 activations + output buffer.
2. **fused kernels** — RMSNorm + matmul, RoPE + attention. reduce dispatch count.
3. **flash attention** — fused QKV with simdgroup MMA. memory savings for long context.
4. **KV cache** — efficient append + lookup for autoregressive generation.
5. **model loader** — read GGUF or safetensors into quantized buffers.
6. **tokenizer** — BPE tokenizer in Rust (or bind sentencepiece).

### E. hardware-specific tuning

- M1 Pro: 16 cores, 200 GB/s, 24MB SLC, 256KB L2
- M2 Pro: same arch, 19 cores, higher clock
- M3 Pro: different arch, mesh shaders, ray tracing units (may affect occupancy)
- M4: new generation — profile when available
- A-series (iPhone): fewer cores, lower BW, but same simdgroup_matrix ISA

## priority order

1. matvec decode optimization (A) — direct tok/s impact
2. inference pipeline (D) — make it usable
3. Q4_K format support (A.5) — llama.cpp compatibility
4. flash attention (D.3) — memory efficiency
5. fp16 last mile (C) — prestige + prefill speed
6. stream-K (B.3) — better GPU utilization on non-square

## metrics to track

- GFLOPS (compute efficiency)
- GB/s (bandwidth efficiency)
- tok/s equivalent (practical output)
- % of hardware ceiling (how close to silicon limits)
- thermal sustained vs peak (real-world performance)
