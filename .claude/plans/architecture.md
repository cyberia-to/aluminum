# rmetal architecture plan — Metal backend for llm runtime

## role in cyb stack

rmetal = the Metal backend. it provides:
- GPU dispatch layer (device, queue, command buffer, encoder)
- pre-compiled jet kernels (matmul, attention, norm, activation, conv)
- tensor abstraction with shape/dtype
- auto-tune dispatch table per hardware
- deterministic execution for STARK provability

```
cyb/graph/llm.md (graph IR + scheduler)
    │
    └── rmetal (Metal backend)
            ├── src/           core FFI (done: device, buffer, command, encoder, pipeline, shader)
            ├── src/tensor.rs  typed tensor with shape/dtype/residency
            ├── src/jet/       pre-compiled jet kernels
            │   ├── matmul.rs  f16, q4, q8, ternary dispatch
            │   ├── attention.rs  flash attention, cross-attention
            │   ├── norm.rs    rmsnorm, layernorm, groupnorm, batchnorm
            │   ├── activation.rs  silu, gelu, relu, sigmoid, softmax
            │   ├── rope.rs    rotary position embedding
            │   ├── conv.rs    conv1d, conv2d, conv3d
            │   ├── kv_cache.rs  append, lookup, paged
            │   ├── embed.rs   token + position embedding
            │   └── quant.rs   quantize/dequantize runtime conversion
            ├── src/registry.rs  jet hash → pipeline mapping
            ├── src/tune.rs    auto-benchmark, dispatch table cache
            └── tools/         benchmarks (current: bench.rs, quant_bench.rs)
```

## tensor abstraction

```rust
pub struct Tensor {
    buffer: MtlBuffer,
    shape: Shape,          // e.g. [512, 4096]
    dtype: DType,          // F16, F32, Q4_0, Q4_K, Q8_0, Ternary
    layout: Layout,        // RowMajor, ColMajor, BlockQ4 { block_size: 32, row_major: bool }
    residency: Residency,  // Resident, Cached, Streamed
}

pub enum DType {
    F16, F32, BF16,
    Q4_0,   // 4.5 bits/weight, 32-element blocks
    Q4_K,   // 4.5 bits/weight, 256-element super-blocks (llama.cpp compat)
    Q8_0,   // 8.5 bits/weight
    Ternary, // 1.58 bits/weight (BitNet: -1, 0, +1)
}
```

## jet interface

```rust
pub trait Jet {
    fn name(&self) -> &str;
    fn hash(&self) -> u64;  // formula hash for STARK
    fn dispatch(
        &self,
        encoder: &MtlComputeEncoder,
        inputs: &[&Tensor],
        output: &Tensor,
        attrs: &Attrs,
    );
}

pub struct JetRegistry {
    jets: HashMap<u64, Box<dyn Jet>>,
    dispatch_table: DispatchTable,  // (jet_hash, shape) → (pipeline, grid, group)
}
```

## implementation phases

### phase 0 — tensor + jet framework (next)
- Tensor struct wrapping MtlBuffer with shape/dtype
- JetRegistry with hash-based lookup
- MatmulJet: auto-dispatch based on input dtypes:
  - (f16, f16) → our fp16 matmul kernel (3863 GFLOPS)
  - (f16, q4_0) → our Q4 matmul kernel (3189 GFLOPS)
  - (f16, q4_0, batch>1) → our batched matvec (718 GFLOPS at batch=8)
- auto crossover: matmul vs batched-matvec based on M dimension

### phase 1 — core LLM jets
- RMSNorm jet (elementwise, trivial kernel)
- SiLU jet (elementwise)
- RoPE jet (per-head rotation)
- Softmax jet (with SIMD reduction)
- Embedding jet (lookup table)
- KV cache jet (append + paged lookup)

### phase 2 — flash attention
- fused QKV matmul + softmax + output matmul
- single kernel dispatch, memory-efficient for long context
- uses simdgroup_multiply_accumulate for QK and PV matmuls

### phase 3 — format loaders
- GGUF parser (read Q4/Q8 tensors + metadata)
- safetensors parser (mmap tensors + config.json)
- ONNX parser (protobuf → graph IR nodes)
- architecture templates: transformer_decoder, encoder, whisper, etc.

### phase 4 — additional dtypes
- Q4_K (256-element super-blocks, llama.cpp compat)
- Q8_0 (8-bit quantization)
- Ternary (BitNet: -1/0/+1, add/subtract matmul — no multiply!)

### phase 5 — auto-tune
- benchmark each jet at various shapes on current hardware
- cache dispatch table to disk
- choose optimal grid/group/tile per (jet, shape, hardware)

### phase 6 — vision/audio jets
- conv2d, conv3d (YOLO, diffusion VAE, video)
- groupnorm (diffusion UNet)
- interpolate, pixel_shuffle
- adaln (DiT models)

## key design decisions

1. **MSL compiled at startup, not runtime** — pre-compile all ~48 jet kernels
   into MTLComputePipelineState at init. no JIT compilation during inference.

2. **row-major Q4 layout as default** — our benchmarks show row-major is
   consistently better for both matmul and matvec. store Q4 weights
   as [K/32][N] not [N][K/32].

3. **deterministic reductions** — for STARK provability, all reduce ops
   use tree reduction with fixed thread mapping. same input → same output
   across runs on same Metal device.

4. **batch-aware dispatch** — matmul for M≥16 (prefill), batched-matvec
   for M=1-8 (decode). crossover point from auto-tune.

5. **zero-copy between jets** — output Tensor of one jet is input to next.
   no CPU roundtrip. all intermediates stay on GPU.

6. **pre-allocated buffer pool** — Tensor buffers allocated once at model
   load. inference reuses buffers via lifetime analysis. no alloc during
   generation.

## metrics from current benchmarks

```
fp16 matmul:        3,863 GFLOPS  (91.6% of 4,218 ceiling)
Q4 matmul:          3,189 GFLOPS
Q4 matvec batch=1:    247 GFLOPS, 29 tok/s
Q4 matvec batch=8:    718 GFLOPS, 83 tok/s
Q4 matvec batch=16:   814 GFLOPS, 94 tok/s
```

these are world-class numbers. the jet framework wraps them
without losing performance.
