//! Encode benchmarks: overhead, unchecked, batch encode, encoder batch

use aluminium::MtlDevice;
use std::time::Instant;

use super::NOOP_SRC;

/// Pure CPU encoding overhead — no per-iter GPU wait.
pub fn encode_overhead(iters: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();

    for _ in 0..10 {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        enc.set_pipeline(&pipe);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((256, 1, 1), (64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let t0 = Instant::now();
    let mut last_cmd = None;
    for _ in 0..iters {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        enc.set_pipeline(&pipe);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((256, 1, 1), (64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        last_cmd = Some(cmd);
    }
    if let Some(cmd) = last_cmd {
        cmd.wait_until_completed();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Pure CPU encoding via unchecked path
pub fn encode_unchecked(iters: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();

    for _ in 0..10 {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        enc.set_pipeline(&pipe);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((256, 1, 1), (64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let t0 = Instant::now();
    let mut last_cmd = None::<aluminium::MtlCommandBuffer>;
    for _ in 0..iters {
        unsafe {
            let cmd = queue.command_buffer_unchecked();
            let enc = cmd.compute_encoder_unchecked();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            last_cmd = Some(cmd);
        }
    }
    if let Some(cmd) = last_cmd {
        cmd.wait_until_completed();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Batch encode via ComputeDispatcher: N dispatches per command buffer
pub fn batch_encode(batch_size: usize, iters: usize) -> f64 {
    use aluminium::ComputeDispatcher;

    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();
    let disp = ComputeDispatcher::new(&queue);

    for _ in 0..3 {
        unsafe {
            disp.dispatch_batch(|batch| {
                for _ in 0..batch_size {
                    batch.set_pipeline(&pipe);
                    batch.set_buffer(&buf, 0, 0);
                    batch.dispatch_threads((256, 1, 1), (64, 1, 1));
                }
            });
        }
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            disp.dispatch_batch(|batch| {
                for _ in 0..batch_size {
                    batch.set_pipeline(&pipe);
                    batch.set_buffer(&buf, 0, 0);
                    batch.dispatch_threads((256, 1, 1), (64, 1, 1));
                }
            });
        }
    }
    t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
}

/// Batch encode via MtlComputeEncoder (msgSend per call)
pub fn encode_encoder(batch_size: usize, iters: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();

    for _ in 0..3 {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        for _ in 0..batch_size {
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        for _ in 0..batch_size {
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
}
