# src/

core library. zero external dependencies — only macOS system frameworks via FFI.

| file | purpose |
|------|---------|
| `lib.rs` | public API surface: re-exports, error types |
| `device.rs` | `MtlDevice` — GPU discovery, properties, factory methods |
| `buffer.rs` | `MtlBuffer` — zero-copy shared CPU↔GPU memory |
| `command.rs` | `MtlCommandQueue`, `MtlCommandBuffer` |
| `encoder.rs` | `MtlComputeEncoder`, `MtlBlitEncoder` |
| `dispatch.rs` | `ComputeDispatcher`, `BatchEncoder`, `GpuFuture` |
| `shader.rs` | `MtlLibrary`, `MtlFunction` — MSL compilation |
| `pipeline.rs` | `MtlComputePipeline`, `MtlRenderPipeline` |
| `texture.rs` | `MtlTexture` |
| `sync.rs` | `MtlFence`, `MtlEvent`, `MtlSharedEvent` |
| `fp16.rs` | fp16↔f32 conversion (NEON + software fallback) |
| `probe.rs` | `metal_probe` binary — 5-level GPU capability probe |
| `tests.rs` | unit tests |
| `ffi/` | raw FFI bindings — see [ffi/README.md](ffi/README.md) |
