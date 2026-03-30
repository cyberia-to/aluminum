//! metal_probe — Metal GPU discovery and capability probe

use aruminium::{MetalError, MtlDevice};

fn main() -> Result<(), MetalError> {
    println!("=== Metal Probe ===\n");

    // Level 1: Device discovery
    println!("--- Level 1: Device Discovery ---");
    let devices = MtlDevice::all()?;
    println!("Found {} Metal device(s)\n", devices.len());

    for (i, dev) in devices.iter().enumerate() {
        println!("Device {}: {}", i, dev.name());
        println!("  Unified memory: {}", dev.has_unified_memory());
        println!(
            "  Max buffer length: {} MB",
            dev.max_buffer_length() / (1024 * 1024)
        );
        let threads = dev.max_threads_per_threadgroup();
        println!(
            "  Max threads/threadgroup: {}x{}x{}",
            threads.width, threads.height, threads.depth
        );
        let wss = dev.recommended_max_working_set_size();
        println!("  Recommended max working set: {} MB", wss / (1024 * 1024));
        println!();
    }

    // Level 2: Buffer creation
    println!("--- Level 2: Buffer Creation ---");
    let device = MtlDevice::system_default()?;
    let buf = device.new_buffer(4096)?;
    println!("Created 4096-byte shared buffer");
    buf.with_data_mut(|d| {
        d[0] = 0xDE;
        d[1] = 0xAD;
    });
    buf.with_data(|d| {
        println!("  Read back: [{:#04x}, {:#04x}]", d[0], d[1]);
    });
    println!();

    // Level 3: Shader compilation
    println!("--- Level 3: Shader Compilation ---");
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void probe_add(device float *a [[buffer(0)]],
                              device float *b [[buffer(1)]],
                              device float *c [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {
            c[id] = a[id] + b[id];
        }
    "#;
    let lib = device.new_library_with_source(source)?;
    println!("Compiled MSL source");
    println!("  Functions: {:?}", lib.function_names());
    let func = lib.get_function("probe_add")?;
    println!("  Got function: {}", func.name());
    println!();

    // Level 4: Compute pipeline
    println!("--- Level 4: Compute Pipeline ---");
    let pipeline = device.new_compute_pipeline(&func)?;
    println!(
        "  Max threads/threadgroup: {}",
        pipeline.max_total_threads_per_threadgroup()
    );
    println!(
        "  Thread execution width: {}",
        pipeline.thread_execution_width()
    );
    println!();

    // Level 5: Compute dispatch
    println!("--- Level 5: Compute Dispatch ---");
    let n = 256usize;
    let buf_a = device.new_buffer(n * 4)?;
    let buf_b = device.new_buffer(n * 4)?;
    let buf_c = device.new_buffer(n * 4)?;

    buf_a.with_f32_mut(|d| {
        for v in d.iter_mut().take(n) {
            *v = 1.0;
        }
    });
    buf_b.with_f32_mut(|d| {
        for v in d.iter_mut().take(n) {
            *v = 2.0;
        }
    });

    let queue = device.new_command_queue()?;
    let cmd = queue.command_buffer()?;
    let enc = cmd.compute_encoder()?;
    enc.set_pipeline(&pipeline);
    enc.set_buffer(&buf_a, 0, 0);
    enc.set_buffer(&buf_b, 0, 1);
    enc.set_buffer(&buf_c, 0, 2);
    enc.dispatch_threads((n, 1, 1), (64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ok = buf_c.with_f32(|d| {
        let mut pass = true;
        for (i, &v) in d.iter().enumerate().take(n) {
            if (v - 3.0).abs() > 1e-6 {
                println!("  FAIL: c[{}] = {} (expected 3.0)", i, v);
                pass = false;
                break;
            }
        }
        pass
    });
    if ok {
        println!("  Compute dispatch: PASS (256 additions verified)");
    }

    println!("\n=== Probe Complete ===");
    Ok(())
}
