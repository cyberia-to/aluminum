//! Vector addition on Metal GPU — minimal compute example

use metal::{MetalError, MtlDevice};

fn main() -> Result<(), MetalError> {
    let device = MtlDevice::system_default()?;
    println!("Device: {}", device.name());

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

    let lib = device.new_library_with_source(source)?;
    let func = lib.get_function("vecadd")?;
    let pipeline = device.new_compute_pipeline(&func)?;

    let n = 1024usize;
    let buf_a = device.new_buffer(n * 4)?;
    let buf_b = device.new_buffer(n * 4)?;
    let buf_c = device.new_buffer(n * 4)?;

    buf_a.with_f32_mut(|d| {
        for i in 0..n {
            d[i] = i as f32;
        }
    });
    buf_b.with_f32_mut(|d| {
        for i in 0..n {
            d[i] = (n - i) as f32;
        }
    });

    let cmd = queue.command_buffer()?;
    let enc = cmd.compute_encoder()?;
    enc.set_pipeline(&pipeline);
    enc.set_buffer(&buf_a, 0, 0);
    enc.set_buffer(&buf_b, 0, 1);
    enc.set_buffer(&buf_c, 0, 2);
    enc.dispatch_threads((n, 1, 1), (pipeline.thread_execution_width(), 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    buf_c.with_f32(|d| {
        let mut ok = true;
        for i in 0..n {
            let expected = n as f32;
            if (d[i] - expected).abs() > 1e-6 {
                println!("FAIL: c[{}] = {} (expected {})", i, d[i], expected);
                ok = false;
                break;
            }
        }
        if ok {
            println!("PASS: {} vector additions verified (all = {})", n, n);
        }
    });

    Ok(())
}
