# guide

practical patterns for using aruminium.

## device setup

```rust
use aruminium::{MtlDevice, MetalError};

let device = MtlDevice::system_default()?;
println!("{}, unified={}, max_buf={}MB",
    device.name(),
    device.has_unified_memory(),
    device.max_buffer_length() / (1024 * 1024));
```

enumerate all GPUs (Mac Pro with multiple GPUs):

```rust
for dev in MtlDevice::all()? {
    println!("{}", dev.name());
}
```

## buffers

### shared (CPU + GPU)

```rust
let buf = device.new_buffer(1024 * 4)?;  // 1024 floats

// write from CPU
buf.with_f32_mut(|data| {
    for (i, v) in data.iter_mut().enumerate() {
        *v = i as f32;
    }
});

// read back after GPU work
buf.with_f32(|data| {
    println!("first: {}", data[0]);
});
```

### private (GPU-only)

```rust
let gpu_buf = device.new_buffer_private(size)?;
assert!(!gpu_buf.is_shared());  // no CPU access

// copy data in via blit
let staging = device.new_buffer_with_data(&bytes)?;
let cmd = queue.command_buffer()?;
let blit = cmd.blit_encoder()?;
blit.copy_buffer(&staging, 0, &gpu_buf, 0, size);
blit.end_encoding();
cmd.commit();
cmd.wait_until_completed();
```

use private buffers for intermediate results that stay on GPU
between kernel dispatches. higher bandwidth than shared.

## compile and dispatch

```rust
let source = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void saxpy(device float *y [[buffer(0)]],
                      device const float *x [[buffer(1)]],
                      constant float &a [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
        y[id] = a * x[id] + y[id];
    }
"#;

let lib = device.new_library_with_source(source)?;
let func = lib.get_function("saxpy")?;
let pipeline = device.new_compute_pipeline(&func)?;

let queue = device.new_command_queue()?;
let cmd = queue.command_buffer()?;
let enc = cmd.compute_encoder()?;

enc.set_pipeline(&pipeline);
enc.set_buffer(&buf_y, 0, 0);
enc.set_buffer(&buf_x, 0, 1);

let alpha: f32 = 2.0;
let alpha_bytes = alpha.to_ne_bytes();
enc.set_bytes(&alpha_bytes, 2);

enc.dispatch_threads((n, 1, 1), (256, 1, 1));
enc.end_encoding();
cmd.commit();
cmd.wait_until_completed();
```

## choosing threadgroup size

query the pipeline for hardware limits:

```rust
let max = pipeline.max_total_threads_per_threadgroup();  // e.g. 1024
let simd = pipeline.thread_execution_width();            // e.g. 32
let tg_mem = pipeline.static_threadgroup_memory_length(); // bytes used
```

rules of thumb:
- 1D work: `(256, 1, 1)` or `(simd, 1, 1)` for small workloads
- 2D work (matmul): `(16, 16, 1)` = 256 threads
- never exceed `max_total_threads_per_threadgroup`
- `dispatch_threads` handles non-uniform grids (n not divisible by group size)
- `dispatch_threadgroups` needs manual `ceil(n / group)` calculation

## GPU timing

```rust
cmd.commit();
cmd.wait_until_completed();
let gpu_ms = cmd.gpu_time() * 1000.0;
println!("GPU: {:.3} ms", gpu_ms);
```

`gpu_start_time()` and `gpu_end_time()` return absolute seconds
since device boot. `gpu_time()` = end - start.

## hot-loop dispatch (inference)

for repeated dispatches (inference decode loop), use `ComputeDispatcher`:

```rust
use aruminium::ComputeDispatcher;

let disp = ComputeDispatcher::new(&queue);

// single dispatch
unsafe {
    disp.dispatch(
        &pipeline,
        &[(&buf_a, 0, 0), (&buf_b, 0, 1), (&buf_c, 0, 2)],
        (n, 1, 1),
        (256, 1, 1),
    );
}
```

### batch dispatch (multiple kernels, one command buffer)

```rust
unsafe {
    disp.dispatch_batch(|batch| {
        // kernel 1: norm
        batch.set_pipeline(&norm_pipe);
        batch.set_buffer(&buf, 0, 0);
        batch.dispatch_threads((hidden, 1, 1), (256, 1, 1));

        // kernel 2: matmul (same encoder, different pipeline)
        batch.set_pipeline(&matmul_pipe);
        batch.set_buffer(&buf, 0, 0);
        batch.set_buffer(&weights, 0, 1);
        batch.set_buffer(&output, 0, 2);
        batch.dispatch_threads((m, n, 1), (16, 16, 1));
    });
}
```

### pipelined dispatch (overlap CPU encoding with GPU execution)

```rust
let mut prev = None;
for layer in 0..num_layers {
    let future = unsafe {
        disp.dispatch_batch_async(|batch| {
            // encode layer N
            batch.set_pipeline(&pipe);
            batch.set_buffer(&bufs[layer], 0, 0);
            batch.dispatch_threads(grid, group);
        })
    };
    if let Some(p) = prev { p.wait(); }
    prev = Some(future);
}
if let Some(p) = prev { p.wait(); }
```

### raw batch (caller manages autorelease pool)

```rust
aruminium::autorelease_pool(|| {
    for step in 0..decode_steps {
        unsafe {
            disp.dispatch_batch_raw(|batch| {
                // all dispatches for one decode step
            });
        }
    }
});
```

one pool for the entire loop instead of per-batch.

## fp16 conversion

```rust
use aruminium::{fp16_to_f32, f32_to_fp16, cvt_f16_f32, cvt_f32_f16};

// single value
let half: u16 = f32_to_fp16(3.14);
let full: f32 = fp16_to_f32(half);

// bulk (NEON-optimized on aarch64)
let src: Vec<u16> = weights_fp16;
let mut dst = vec![0.0f32; src.len()];
cvt_f16_f32(&mut dst, &src);
```

## error handling

all fallible operations return `Result<T, MetalError>`. error variants
carry context strings from Metal.framework (e.g. shader compilation
errors include the MSL compiler diagnostic).

```rust
match device.new_library_with_source(bad_source) {
    Err(MetalError::LibraryCompilationFailed(msg)) => {
        eprintln!("MSL error: {}", msg);
    }
    _ => {}
}
```
