//! Basic benchmarks: device, buffer, shader, large compute, inference sim

use aluminium::MtlDevice;
use std::time::Instant;

use super::SAXPY_SRC;

pub fn device_discovery(iters: usize) -> f64 {
    let _ = MtlDevice::system_default().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = MtlDevice::system_default().unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn buffer_creation(iters: usize, size: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    for _ in 0..3 {
        let _ = dev.new_buffer(size).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.new_buffer(size).unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn shader_compile(iters: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    let src = SAXPY_SRC;
    let _ = dev.new_library_with_source(src).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = dev.new_library_with_source(src).unwrap();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

pub fn large_compute(iters: usize, n: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(SAXPY_SRC).unwrap();
    let func = lib.get_function("saxpy").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();

    let buf_x = dev.new_buffer(n * 4).unwrap();
    let buf_y = dev.new_buffer(n * 4).unwrap();
    buf_x.with_f32_mut(|d| {
        for (i, v) in d.iter_mut().enumerate() {
            *v = i as f32;
        }
    });
    buf_y.with_f32_mut(|d| d.fill(1.0));
    let a_val: f32 = 2.0;
    let a_bytes = a_val.to_le_bytes();

    for _ in 0..3 {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        enc.set_pipeline(&pipe);
        enc.set_buffer(&buf_x, 0, 0);
        enc.set_buffer(&buf_y, 0, 1);
        enc.set_bytes(&a_bytes, 2);
        enc.dispatch_threads((n, 1, 1), (256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    buf_y.with_f32_mut(|d| d.fill(1.0));

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        enc.set_pipeline(&pipe);
        enc.set_buffer(&buf_x, 0, 0);
        enc.set_buffer(&buf_y, 0, 1);
        enc.set_bytes(&a_bytes, 2);
        enc.dispatch_threads((n, 1, 1), (256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// SAXPY with ComputeDispatcher
pub fn large_compute_imp(iters: usize, n: usize) -> f64 {
    use aluminium::ComputeDispatcher;

    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(SAXPY_SRC).unwrap();
    let func = lib.get_function("saxpy").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf_x = dev.new_buffer(n * 4).unwrap();
    let buf_y = dev.new_buffer(n * 4).unwrap();
    buf_x.with_f32_mut(|d| {
        for (i, v) in d.iter_mut().enumerate() {
            *v = i as f32;
        }
    });
    buf_y.with_f32_mut(|d| d.fill(1.0));
    let a: f32 = 2.0;
    let ab = a.to_le_bytes();
    let disp = ComputeDispatcher::new(&queue);

    for _ in 0..3 {
        unsafe {
            disp.dispatch_with_bytes(
                &pipe,
                &[(&buf_x, 0, 0), (&buf_y, 0, 1)],
                &ab,
                2,
                (n, 1, 1),
                (256, 1, 1),
            );
        }
    }
    buf_y.with_f32_mut(|d| d.fill(1.0));

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            disp.dispatch_with_bytes(
                &pipe,
                &[(&buf_x, 0, 0), (&buf_y, 0, 1)],
                &ab,
                2,
                (n, 1, 1),
                (256, 1, 1),
            );
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Simulates inference pass: 3 kernels x N layers, single batch.
pub fn inference_sim(layers: usize, iters: usize) -> f64 {
    use aluminium::ComputeDispatcher;

    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void matmul_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                             device float *c [[buffer(2)]], uint id [[thread_position_in_grid]]) {
            c[id] = a[id] * b[id];
        }
        kernel void relu_k(device float *x [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            x[id] = max(x[id], 0.0f);
        }
        kernel void add_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + b[id];
        }
    "#;
    let lib = dev.new_library_with_source(src).unwrap();
    let matmul = dev
        .new_compute_pipeline(&lib.get_function("matmul_k").unwrap())
        .unwrap();
    let relu = dev
        .new_compute_pipeline(&lib.get_function("relu_k").unwrap())
        .unwrap();
    let add = dev
        .new_compute_pipeline(&lib.get_function("add_k").unwrap())
        .unwrap();
    let n = 4096usize;
    let buf_a = dev.new_buffer(n * 4).unwrap();
    let buf_b = dev.new_buffer(n * 4).unwrap();
    let buf_c = dev.new_buffer(n * 4).unwrap();
    let disp = ComputeDispatcher::new(&queue);

    for _ in 0..3 {
        unsafe {
            disp.dispatch_batch(|b| {
                for _ in 0..layers {
                    b.set_pipeline(&matmul);
                    b.set_buffer(&buf_a, 0, 0);
                    b.set_buffer(&buf_b, 0, 1);
                    b.set_buffer(&buf_c, 0, 2);
                    b.dispatch_threads((n, 1, 1), (256, 1, 1));
                    b.set_pipeline(&relu);
                    b.set_buffer(&buf_c, 0, 0);
                    b.dispatch_threads((n, 1, 1), (256, 1, 1));
                    b.set_pipeline(&add);
                    b.set_buffer(&buf_a, 0, 0);
                    b.set_buffer(&buf_c, 0, 1);
                    b.dispatch_threads((n, 1, 1), (256, 1, 1));
                }
            });
        }
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            disp.dispatch_batch(|b| {
                for _ in 0..layers {
                    b.set_pipeline(&matmul);
                    b.set_buffer(&buf_a, 0, 0);
                    b.set_buffer(&buf_b, 0, 1);
                    b.set_buffer(&buf_c, 0, 2);
                    b.dispatch_threads((n, 1, 1), (256, 1, 1));
                    b.set_pipeline(&relu);
                    b.set_buffer(&buf_c, 0, 0);
                    b.dispatch_threads((n, 1, 1), (256, 1, 1));
                    b.set_pipeline(&add);
                    b.set_buffer(&buf_a, 0, 0);
                    b.set_buffer(&buf_c, 0, 1);
                    b.dispatch_threads((n, 1, 1), (256, 1, 1));
                }
            });
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}
