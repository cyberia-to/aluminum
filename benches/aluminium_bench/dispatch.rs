//! Dispatch benchmarks: overhead, unchecked, autoreleased, raw msgSend, IMP, pipelined

use aluminium::MtlDevice;
use std::time::Instant;

use super::NOOP_SRC;

pub fn dispatch_overhead(iters: usize) -> f64 {
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
    for _ in 0..iters {
        let cmd = queue.command_buffer().unwrap();
        let enc = cmd.compute_encoder().unwrap();
        enc.set_pipeline(&pipe);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((256, 1, 1), (64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Unchecked dispatch — zero-overhead wrappers, unretained references
pub fn dispatch_unchecked(iters: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();

    for _ in 0..10 {
        unsafe {
            let cmd = queue.command_buffer_unchecked();
            let enc = cmd.compute_encoder_unchecked();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            let cmd = queue.command_buffer_unchecked();
            let enc = cmd.compute_encoder_unchecked();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Autoreleased dispatch — no retain/release at all
pub fn dispatch_autoreleased(iters: usize) -> f64 {
    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();

    for _ in 0..10 {
        aluminium::autorelease_pool(|| unsafe {
            let cmd = queue.command_buffer_autoreleased();
            let enc = cmd.compute_encoder_autoreleased();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
    }

    let t0 = Instant::now();
    aluminium::autorelease_pool(|| {
        for _ in 0..iters {
            unsafe {
                let cmd = queue.command_buffer_autoreleased();
                let enc = cmd.compute_encoder_autoreleased();
                enc.set_pipeline(&pipe);
                enc.set_buffer(&buf, 0, 0);
                enc.dispatch_threads((256, 1, 1), (64, 1, 1));
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }
    });
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Raw dispatch — bypass all wrappers. Pure objc_msgSend + cached sel.
pub fn dispatch_raw(iters: usize) -> f64 {
    use aluminium::ffi::*;
    use std::ffi::c_void;

    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();

    let q = queue.as_raw();
    let p = pipe.as_raw();
    let b = buf.as_raw();
    let s_cb = SEL_commandBuffer();
    let s_ce = SEL_computeCommandEncoder();
    let s_sp = SEL_setComputePipelineState();
    let s_sb = SEL_setBuffer_offset_atIndex();
    let s_dt = SEL_dispatchThreads();
    let s_ee = SEL_endEncoding();
    let s_cm = SEL_commit();
    let s_wt = SEL_waitUntilCompleted();

    type F0 = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    type FV = unsafe extern "C" fn(ObjcId, ObjcSel);
    type F1 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
    type F3 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger, NSUInteger);
    type FD = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);

    let f0: F0 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let fv: FV = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let f1: F1 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let f3: F3 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
    let fd: FD = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };

    let grid = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: 64,
        height: 1,
        depth: 1,
    };

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
    for _ in 0..iters {
        unsafe {
            let cmd = f0(q, s_cb);
            objc_retainAutoreleasedReturnValue(cmd);
            let enc = f0(cmd, s_ce);
            objc_retainAutoreleasedReturnValue(enc);
            f1(enc, s_sp, p);
            f3(enc, s_sb, b, 0, 0);
            fd(enc, s_dt, grid, group);
            fv(enc, s_ee);
            fv(cmd, s_cm);
            fv(cmd, s_wt);
            objc_release(enc);
            objc_release(cmd);
        }
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// ComputeDispatcher: pre-resolved IMP
pub fn dispatch_imp(iters: usize) -> f64 {
    use aluminium::ComputeDispatcher;

    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();
    let disp = ComputeDispatcher::new(&queue);

    for _ in 0..10 {
        unsafe { disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1)) };
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1)) };
    }
    t0.elapsed().as_secs_f64() / iters as f64
}

/// Pipelined dispatch: overlap GPU execution of batch N with encoding of N+1
pub fn dispatch_pipelined(iters: usize) -> f64 {
    use aluminium::ComputeDispatcher;

    let dev = MtlDevice::system_default().unwrap();
    let queue = dev.new_command_queue().unwrap();
    let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
    let func = lib.get_function("noop").unwrap();
    let pipe = dev.new_compute_pipeline(&func).unwrap();
    let buf = dev.new_buffer(256 * 4).unwrap();
    let disp = ComputeDispatcher::new(&queue);

    for _ in 0..10 {
        unsafe {
            disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1));
        }
    }

    let t0 = Instant::now();
    let mut prev = None;
    for _ in 0..iters {
        let future = unsafe {
            disp.dispatch_batch_async(|batch| {
                batch.set_pipeline(&pipe);
                batch.set_buffer(&buf, 0, 0);
                batch.dispatch_threads((256, 1, 1), (64, 1, 1));
            })
        };
        if let Some(p) = prev {
            aluminium::GpuFuture::wait(p);
        }
        prev = Some(future);
    }
    if let Some(p) = prev {
        aluminium::GpuFuture::wait(p);
    }
    t0.elapsed().as_secs_f64() / iters as f64
}
