//! Integration tests — full GPU pipeline: compile → dispatch → verify

use aruminium::{autorelease_pool, Dispatch, Gpu, GpuError};

#[test]
fn vecadd_1024() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void vecadd(device float *a [[buffer(0)]],
                           device float *b [[buffer(1)]],
                           device float *c [[buffer(2)]],
                           uint id [[thread_position_in_grid]]) {
            c[id] = a[id] + b[id];
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("vecadd")?)?;

    let n = 1024usize;
    let buf_a = device.buffer(n * 4)?;
    let buf_b = device.buffer(n * 4)?;
    let buf_c = device.buffer(n * 4)?;

    buf_a.write_f32(|d| {
        for i in 0..n {
            d[i] = i as f32;
        }
    });
    buf_b.write_f32(|d| {
        for i in 0..n {
            d[i] = (n - i) as f32;
        }
    });

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf_a, 0, 0);
    enc.bind_buffer(&buf_b, 0, 1);
    enc.bind_buffer(&buf_c, 0, 2);
    enc.launch((n, 1, 1), (64, 1, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    buf_c.read_f32(|d| {
        for i in 0..n {
            assert!(
                (d[i] - n as f32).abs() < 1e-6,
                "vecadd mismatch at {}: got {}",
                i,
                d[i]
            );
        }
    });
    Ok(())
}

#[test]
fn matmul_64x64() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint M; uint N; uint K; };
        kernel void matmul(device const float *A [[buffer(0)]],
                           device const float *B [[buffer(1)]],
                           device float *C       [[buffer(2)]],
                           constant P &p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
            uint row = gid.y, col = gid.x;
            if (row >= p.M || col >= p.N) return;
            float sum = 0.0;
            for (uint k = 0; k < p.K; k++)
                sum += A[row * p.K + k] * B[k * p.N + col];
            C[row * p.N + col] = sum;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("matmul")?)?;

    let m = 64usize;
    let buf_a = device.buffer(m * m * 4)?;
    let buf_b = device.buffer(m * m * 4)?;
    let buf_c = device.buffer(m * m * 4)?;

    // A = identity, B = ones => C = ones
    buf_a.write_f32(|d| {
        for i in 0..m {
            for j in 0..m {
                d[i * m + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    });
    buf_b.write_f32(|d| d.fill(1.0));

    #[repr(C)]
    struct P {
        m: u32,
        n: u32,
        k: u32,
    }
    let params = P {
        m: m as u32,
        n: m as u32,
        k: m as u32,
    };
    let params_bytes = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let cmd = queue.commands()?;
    let enc = cmd.encoder()?;
    enc.bind(&pipe);
    enc.bind_buffer(&buf_a, 0, 0);
    enc.bind_buffer(&buf_b, 0, 1);
    enc.bind_buffer(&buf_c, 0, 2);
    enc.push(params_bytes, 3);
    enc.launch((m, m, 1), (16, 16, 1));
    enc.finish();
    cmd.submit();
    cmd.wait();

    buf_c.read_f32(|d| {
        let max_err: f32 = d.iter().map(|&v| (v - 1.0).abs()).fold(0.0, f32::max);
        assert!(max_err < 1e-5, "matmul max_err={}", max_err);
    });
    Ok(())
}

#[test]
fn compute_dispatcher_batch() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void inc(device float *a [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("inc")?)?;

    let n = 256usize;
    let buf = device.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    let disp = Dispatch::new(&queue);

    // batch 10 increments into one command buffer
    unsafe {
        disp.batch(|batch| {
            for _ in 0..10 {
                batch.bind(&pipe);
                batch.bind_buffer(&buf, 0, 0);
                batch.launch((n, 1, 1), (64, 1, 1));
            }
        });
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 10.0).abs() < 1e-6,
                "batch dispatch: d[{}]={}, expected 10.0",
                i,
                v
            );
        }
    });
    Ok(())
}

#[test]
fn compute_dispatcher_async_pipelining() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void inc(device float *a [[buffer(0)]],
                        uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("inc")?)?;

    let n = 256usize;
    let buf = device.buffer(n * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    let disp = Dispatch::new(&queue);

    // pipeline 5 async batches
    let mut prev = None;
    for _ in 0..5 {
        let future = unsafe {
            disp.batch_async(|batch| {
                batch.bind(&pipe);
                batch.bind_buffer(&buf, 0, 0);
                batch.launch((n, 1, 1), (64, 1, 1));
            })
        };
        if let Some(p) = prev {
            aruminium::GpuFuture::wait(p);
        }
        prev = Some(future);
    }
    if let Some(p) = prev {
        aruminium::GpuFuture::wait(p);
    }

    buf.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-6,
                "pipelined: d[{}]={}, expected 5.0",
                i,
                v
            );
        }
    });
    Ok(())
}

#[test]
fn private_buffer_blit_copy() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let n = 256usize;
    let shared = device.buffer(n * 4)?;
    let private = device.buffer_private(n * 4)?;
    let readback = device.buffer(n * 4)?;

    shared.write_f32(|d| {
        for (i, v) in d.iter_mut().enumerate() {
            *v = i as f32;
        }
    });

    // shared -> private via blit
    let cmd = queue.commands()?;
    let blit = cmd.copier()?;
    blit.copy(&shared, 0, &private, 0, n * 4);
    blit.finish();
    cmd.submit();
    cmd.wait();

    // private -> readback via blit
    let cmd = queue.commands()?;
    let blit = cmd.copier()?;
    blit.copy(&private, 0, &readback, 0, n * 4);
    blit.finish();
    cmd.submit();
    cmd.wait();

    readback.read_f32(|d| {
        for (i, &v) in d.iter().enumerate() {
            assert_eq!(v, i as f32, "blit copy mismatch at {}", i);
        }
    });
    Ok(())
}

#[test]
fn autorelease_pool_dispatch() -> Result<(), GpuError> {
    let device = Gpu::open()?;
    let queue = device.new_command_queue()?;

    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.compile(source)?;
    let pipe = device.pipeline(&lib.function("noop")?)?;
    let buf = device.buffer(256 * 4)?;
    buf.write_f32(|d| d.fill(0.0));

    // 100 dispatches in autorelease pool with autoreleased command buffers
    autorelease_pool(|| {
        for _ in 0..100 {
            unsafe {
                let cmd = queue.commands_autoreleased();
                let enc = cmd.encoder_autoreleased();
                enc.bind(&pipe);
                enc.bind_buffer(&buf, 0, 0);
                enc.launch((256, 1, 1), (64, 1, 1));
                enc.finish();
                cmd.submit();
                cmd.wait();
            }
        }
    });

    buf.read_f32(|d| {
        assert!(
            (d[0] - 100.0).abs() < 1e-6,
            "autorelease pool: d[0]={}, expected 100.0",
            d[0]
        );
    });
    Ok(())
}
