# aluminium

the lightest metal.

pure Rust Apple Metal GPU driver. zero external dependencies. direct `objc_msgSend` FFI to Metal.framework — no objc runtime, no Swift, no headers.

```rust
let device = aluminium::MtlDevice::system_default()?;
let queue = device.new_command_queue()?;

let lib = device.new_library_with_source(r#"
    #include <metal_stdlib>
    kernel void add(device float *a [[buffer(0)]],
                    device float *b [[buffer(1)]],
                    device float *c [[buffer(2)]],
                    uint id [[thread_position_in_grid]]) {
        c[id] = a[id] + b[id];
    }
"#)?;

let pipe = device.new_compute_pipeline(&lib.get_function("add")?)?;
let buf = device.new_buffer(n * 4)?;

let cmd = queue.command_buffer()?;
let enc = cmd.compute_encoder()?;
enc.set_pipeline(&pipe);
enc.set_buffer(&buf, 0, 0);
enc.dispatch_threads((n, 1, 1), (256, 1, 1));
enc.end_encoding();
cmd.commit();
cmd.wait_until_completed();
```

## numbers

M1 Pro 16-core:

```
buffer create (1 MB):     0.01 ms
shader compile:           0.01 ms
dispatch overhead:        0.21 ms CPU / 0.18 ms GPU
SAXPY 16M floats:         143 GB/s
fp16 conversion:          58-72 GB/s
```

vs objc2-metal (standard Rust Metal binding):

```
batch encode:    1.13x faster
pipelined:       1.79x faster
inference sim:   1.11x faster (300 layers)
SAXPY:           143 vs 140 GB/s
```

same GPU, same work. aluminium is lighter.

## api

```rust
// device
MtlDevice::system_default() -> Result<MtlDevice>
device.name() -> String
device.has_unified_memory() -> bool

// buffers
device.new_buffer(bytes) -> Result<MtlBuffer>
device.new_buffer_with_data(&[u8]) -> Result<MtlBuffer>
buf.with_data(|&[u8]|)
buf.with_f32_mut(|&mut [f32]|)

// shaders
device.new_library_with_source(&str) -> Result<MtlLibrary>
lib.get_function(&str) -> Result<MtlFunction>
device.new_compute_pipeline(&MtlFunction) -> Result<MtlComputePipeline>

// dispatch
queue.command_buffer() -> Result<MtlCommandBuffer>
cmd.compute_encoder() -> Result<MtlComputeEncoder>
enc.set_pipeline(&pipe)
enc.set_buffer(&buf, offset, index)
enc.dispatch_threadgroups(grid, group)
enc.end_encoding()
cmd.commit()
cmd.wait_until_completed()

// profiling
cmd.gpu_time() -> f64
pipeline.static_threadgroup_memory_length() -> usize

// fp16
aluminium::f32_to_fp16(f32) -> u16
aluminium::fp16_to_f32(u16) -> f32
```

## build

```
cargo build --release
cargo run --example vecadd
cargo run --release -p metal-benches --bin bench
```

requires macOS with Metal-capable GPU.

## architecture

```text
cyb/llm          inference runtime — models, graphs, scheduling
cyb/llm/backend  jets — MSL kernels dispatched via aluminium
aluminium          this crate — device, buffer, pipeline, dispatch
```

aluminium does not depend on cyb/llm. cyb/llm depends on aluminium. one way.

## license

don't trust. don't fear. don't beg.
