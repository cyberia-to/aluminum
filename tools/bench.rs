//! aluminium driver benchmark — buffer, dispatch, fp16 conversion
//!
//! Layer 1 only: measures hardware driver performance.
//! No matmul, no attention, no model knowledge.

use aluminium::{MetalError, MtlDevice};
use std::time::Instant;

fn main() -> Result<(), MetalError> {
    let device = MtlDevice::system_default()?;
    println!("Device: {}", device.name());
    println!("Unified memory: {}", device.has_unified_memory());
    println!(
        "Max buffer: {} MB",
        device.max_buffer_length() / (1024 * 1024)
    );
    println!();

    let queue = device.new_command_queue()?;

    // ── Buffer creation throughput ──
    let sizes = [1024, 1024 * 1024, 64 * 1024 * 1024];
    for &size in &sizes {
        let t0 = Instant::now();
        let _buf = device.new_buffer(size)?;
        let dt = t0.elapsed();
        println!(
            "Buffer {} MB: {:.2} ms",
            size / (1024 * 1024),
            dt.as_secs_f64() * 1000.0
        );
    }
    println!();

    // ── Compute dispatch latency ──
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.new_library_with_source(source)?;
    let func = lib.get_function("noop")?;
    let pipeline = device.new_compute_pipeline(&func)?;

    println!(
        "Pipeline: max_threads={}, simd_width={}, TG_mem={}",
        pipeline.max_total_threads_per_threadgroup(),
        pipeline.thread_execution_width(),
        pipeline.static_threadgroup_memory_length(),
    );

    let n = 1024 * 1024usize;
    let buf = device.new_buffer(n * 4)?;
    buf.with_f32_mut(|d| {
        for v in d.iter_mut().take(n) {
            *v = 0.0;
        }
    });

    // Warmup
    for _ in 0..3 {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipeline);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((n, 1, 1), (256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark: CPU time vs GPU time
    let iters = 100;
    let mut gpu_total = 0.0f64;
    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipeline);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((n, 1, 1), (256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        gpu_total += cmd.gpu_time();
    }
    let cpu_total = t0.elapsed().as_secs_f64();
    let cpu_per = cpu_total / iters as f64;
    let gpu_per = gpu_total / iters as f64;
    let bandwidth = (n * 4 * 2) as f64 / gpu_per / 1e9;
    println!(
        "Dispatch ({} floats): CPU {:.3} ms | GPU {:.3} ms | overhead {:.3} ms",
        n,
        cpu_per * 1000.0,
        gpu_per * 1000.0,
        (cpu_per - gpu_per) * 1000.0,
    );
    println!("Effective bandwidth: {:.1} GB/s", bandwidth);
    println!();

    // ── fp16 conversion benchmark ──
    let n16 = 16 * 1024 * 1024;
    let src16: Vec<u16> = (0..n16)
        .map(|i| aluminium::f32_to_fp16(i as f32 * 0.001))
        .collect();
    let mut dst32 = vec![0.0f32; n16];
    let mut dst16 = vec![0u16; n16];
    let src32: Vec<f32> = (0..n16).map(|i| i as f32 * 0.001).collect();

    aluminium::cvt_f16_f32(&mut dst32, &src16);
    aluminium::cvt_f32_f16(&mut dst16, &src32);

    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        aluminium::cvt_f16_f32(&mut dst32, &src16);
    }
    let dt = t0.elapsed();
    let bw = (n16 as f64 * (2 + 4) as f64 * iters as f64) / dt.as_secs_f64() / 1e9;
    println!(
        "fp16→f32 ({}M): {:.2} ms/iter, {:.1} GB/s",
        n16 / 1_000_000,
        dt.as_secs_f64() * 1000.0 / iters as f64,
        bw
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        aluminium::cvt_f32_f16(&mut dst16, &src32);
    }
    let dt = t0.elapsed();
    let bw = (n16 as f64 * (4 + 2) as f64 * iters as f64) / dt.as_secs_f64() / 1e9;
    println!(
        "f32→fp16 ({}M): {:.2} ms/iter, {:.1} GB/s",
        n16 / 1_000_000,
        dt.as_secs_f64() * 1000.0 / iters as f64,
        bw
    );

    Ok(())
}
