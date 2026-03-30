# moonshots — status

ALL ideas tested except #6 (MPP — researching).

## results

| # | idea | result | GFLOPS | vs baseline |
|---|------|--------|--------|-------------|
| #1 | pipelined MMA + double buf | ✗ worse | 2839 | -19% |
| #2 | async threadgroup copy | ✗ N/A | — | stride mismatch |
| #3 | stream-K | ✗ worse | 2761 | -22% |
| #4 | dual accumulator (8 acc/sg) | ✗ worse | 3073 | -12% |
| #5 | SLC priming | ✓ marginal | +2.6% | cold→warm |
| #6 | Metal Performance Primitives | ✓ exists, needs Metal 4 SDK | — | future benchmark target |
| #7 | Z-curve tile affinity | ✗ same | 3463 | -0.8% |
| #8 | 3×2 accumulator grid | ✗ worse | 2535 | -27% |
| #9 | mixed-precision 2-pass | ✗ N/A | — | already optimal |
| #10 | parallel sub-matmuls | ✗ same | 1.04× | no overlap |
| #11 | direct device load | ✗ worse | 1780 | -49% |

## profiler discoveries

| finding | impact |
|---------|--------|
| GPU time > CPU time by 5% | true perf = 3636, not 3456 |
| TG memory = 8960B → 3 TG/core | occupancy limiter |
| pad+1/+2 = 8448B → still 3 TG/core but 3708 GFLOPS | +1.3% |
| pad+0/+0 = 4 TG/core but bank conflicts | -3% |
| cmd buffer overhead = 0.2ms fixed | 32% at 512, 5% at 2048 |
| more tiles/core = better (86>64>48) | large matrices win |

## additional tests

| test | result |
|------|--------|
| extract_bits() intrinsic | ✗ not faster than mask+shift |
| SIMD shuffle dequant | ✗ no sharing in batch=1 |
| pre-dequant fp16 cache | ✗ 3.6× more data |
| LUT dequant | ✗ bank conflicts |
| BK=64 Q4 matmul | ✗ occupancy loss |
| persistent kernel | ✗ Apple scheduler better |
| cooperative X load | ✗ barriers > redundant reads |
| 1024-thread matvec | ✗ too few threadgroups |
| 32B-padded Q4 blocks | ✗ 44% BW waste |
| batched decode | ✓ 714 GFLOPS batch=8 |
| larger matrices | ✓ 3597 sustained at 8192 |

## current best numbers (GPU timestamps)

```
fp16 matmul 2048 (GPU time):   3708 GFLOPS  (87.9% of 4218 ceiling)
fp16 matmul 8192 (CPU time):   3597 GFLOPS  (85.3% sustained)
Q4 matmul:                     3204 GFLOPS
Q4 matvec batch=1:              242 GFLOPS
Q4 matvec batch=8:              714 GFLOPS, 83 tok/s
```
