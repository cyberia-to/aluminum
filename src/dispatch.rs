//! ComputeDispatcher: pre-resolved IMP dispatch engine for hot loops
//!
//! Resolves all ObjC method implementations at construction time.
//! Hot path calls go through direct function pointers — no objc_msgSend,
//! no selector lookup, no method cache check.

use crate::buffer::MtlBuffer;
use crate::command::MtlCommandQueue;
use crate::ffi::*;
use crate::pipeline::MtlComputePipeline;
use std::ffi::c_void;

// IMP type aliases for the hot path
type ImpId = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
type ImpVoid = unsafe extern "C" fn(ObjcId, ObjcSel);
type ImpObj = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
type ImpBuf = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger, NSUInteger);
type ImpBytes = unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, NSUInteger, NSUInteger);
type ImpDisp = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);

/// # Safety
/// T must be a function pointer type with the same size as `*const c_void`.
unsafe fn resolve_imp<T>(cls: ObjcClass, sel: ObjcSel) -> T {
    assert!(
        std::mem::size_of::<T>() == std::mem::size_of::<*const c_void>(),
        "resolve_imp: T must be pointer-sized"
    );
    std::mem::transmute_copy(&class_getMethodImplementation(cls, sel))
}

/// Pre-resolved compute dispatch engine.
///
/// Resolves all ObjC method implementations once at construction.
/// Every dispatch call goes through a direct function pointer — bypasses
/// objc_msgSend entirely. For inference loops doing thousands of dispatches,
/// this eliminates the selector lookup overhead.
///
/// Uses `commandBufferWithUnretainedReferences` — Metal skips internal
/// retain/release on all resources, saving ~2μs per command buffer.
pub struct ComputeDispatcher {
    queue: ObjcId,
    // Pre-resolved selectors (still needed as IMP arg)
    s_cb: ObjcSel,
    s_ce: ObjcSel,
    s_sp: ObjcSel,
    s_sb: ObjcSel,
    s_by: ObjcSel,
    s_dt: ObjcSel,
    s_dg: ObjcSel,
    s_ee: ObjcSel,
    s_cm: ObjcSel,
    s_wt: ObjcSel,
    // Pre-resolved IMPs — all resolved eagerly in new()
    imp_cb: ImpId,
    imp_ce: ImpId,
    imp_cm: ImpVoid,
    imp_wt: ImpVoid,
    imp_sp: ImpObj,
    imp_sb: ImpBuf,
    imp_by: ImpBytes,
    imp_dt: ImpDisp,
    imp_dg: ImpDisp,
    imp_ee: ImpVoid,
}

impl ComputeDispatcher {
    /// Create a new dispatcher from a command queue.
    /// Resolves all ObjC method implementations eagerly.
    /// Retains the queue — safe to drop the original `MtlCommandQueue`.
    pub fn new(queue: &MtlCommandQueue) -> Self {
        let q = queue.as_raw();
        unsafe { objc_retain(q) };

        // Selectors
        // Use regular commandBuffer — ARC fast-retain skips autorelease round-trip.
        // commandBufferWithUnretainedReferences has different Metal-internal path
        // that can be slower with ARC retain pattern.
        let s_cb = SEL_commandBuffer();
        let s_ce = SEL_computeCommandEncoder();
        let s_sp = SEL_setComputePipelineState();
        let s_sb = SEL_setBuffer_offset_atIndex();
        let s_by = SEL_setBytes_length_atIndex();
        let s_dt = SEL_dispatchThreads();
        let s_dg = SEL_dispatchThreadgroups();
        let s_ee = SEL_endEncoding();
        let s_cm = SEL_commit();
        let s_wt = SEL_waitUntilCompleted();

        unsafe {
            // Resolve queue IMP
            let q_cls = object_getClass(q);
            let imp_cb: ImpId = resolve_imp(q_cls, s_cb);

            // Create a temporary command buffer to get its class
            let cmd = imp_cb(q, s_cb);
            objc_retain(cmd);
            let cmd_cls = object_getClass(cmd);

            let imp_ce: ImpId = resolve_imp(cmd_cls, s_ce);
            let imp_cm: ImpVoid = resolve_imp(cmd_cls, s_cm);
            let imp_wt: ImpVoid = resolve_imp(cmd_cls, s_wt);

            // Create a temporary encoder to get its class
            let enc = imp_ce(cmd, s_ce);
            objc_retain(enc);
            let enc_cls = object_getClass(enc);

            let imp_sp: ImpObj = resolve_imp(enc_cls, s_sp);
            let imp_sb: ImpBuf = resolve_imp(enc_cls, s_sb);
            let imp_by: ImpBytes = resolve_imp(enc_cls, s_by);
            let imp_dt: ImpDisp = resolve_imp(enc_cls, s_dt);
            let imp_dg: ImpDisp = resolve_imp(enc_cls, s_dg);
            let imp_ee: ImpVoid = resolve_imp(enc_cls, s_ee);

            // Cleanup temp objects
            imp_ee(enc, s_ee);
            imp_cm(cmd, s_cm);
            imp_wt(cmd, s_wt);
            objc_release(enc);
            objc_release(cmd);

            ComputeDispatcher {
                queue: q,
                s_cb,
                s_ce,
                s_sp,
                s_sb,
                s_by,
                s_dt,
                s_dg,
                s_ee,
                s_cm,
                s_wt,
                imp_cb,
                imp_ce,
                imp_cm,
                imp_wt,
                imp_sp,
                imp_sb,
                imp_by,
                imp_dt,
                imp_dg,
                imp_ee,
            }
        }
    }

    /// Dispatch a single compute operation. Creates a command buffer,
    /// encodes, commits, and waits for completion.
    ///
    /// Hybrid path: msgSend + ARC fast-retain for cmd/encoder creation,
    /// pre-resolved IMP for all encoding operations.
    ///
    /// # Safety
    /// All buffers and pipeline must remain alive until completion.
    #[inline(always)]
    pub unsafe fn dispatch(
        &self,
        pipeline: &MtlComputePipeline,
        buffers: &[(&MtlBuffer, usize, usize)],
        grid: (usize, usize, usize),
        group: (usize, usize, usize),
    ) {
        // ARC fast-retain: runtime skips autorelease+retain round-trip
        let cmd = msg0_retained(self.queue, self.s_cb);
        let enc = msg0_retained(cmd, self.s_ce);

        (self.imp_sp)(enc, self.s_sp, pipeline.as_raw());
        for &(buf, offset, index) in buffers {
            (self.imp_sb)(enc, self.s_sb, buf.as_raw(), offset, index);
        }
        let g = MTLSize {
            width: grid.0,
            height: grid.1,
            depth: grid.2,
        };
        let t = MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        };
        (self.imp_dt)(enc, self.s_dt, g, t);
        (self.imp_ee)(enc, self.s_ee);
        (self.imp_cm)(cmd, self.s_cm);
        (self.imp_wt)(cmd, self.s_wt);

        objc_release(enc);
        objc_release(cmd);
    }

    /// Dispatch with inline bytes (e.g. uniforms).
    ///
    /// # Safety
    /// All buffers and pipeline must remain alive until completion.
    #[inline(always)]
    pub unsafe fn dispatch_with_bytes(
        &self,
        pipeline: &MtlComputePipeline,
        buffers: &[(&MtlBuffer, usize, usize)],
        bytes: &[u8],
        bytes_index: usize,
        grid: (usize, usize, usize),
        group: (usize, usize, usize),
    ) {
        let cmd = msg0_retained(self.queue, self.s_cb);
        let enc = msg0_retained(cmd, self.s_ce);

        (self.imp_sp)(enc, self.s_sp, pipeline.as_raw());
        for &(buf, offset, index) in buffers {
            (self.imp_sb)(enc, self.s_sb, buf.as_raw(), offset, index);
        }
        (self.imp_by)(
            enc,
            self.s_by,
            bytes.as_ptr() as *const c_void,
            bytes.len(),
            bytes_index,
        );
        let g = MTLSize {
            width: grid.0,
            height: grid.1,
            depth: grid.2,
        };
        let t = MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        };
        (self.imp_dt)(enc, self.s_dt, g, t);
        (self.imp_ee)(enc, self.s_ee);
        (self.imp_cm)(cmd, self.s_cm);
        (self.imp_wt)(cmd, self.s_wt);

        objc_release(enc);
        objc_release(cmd);
    }

    /// Encode multiple dispatches into a single command buffer.
    /// Much more efficient than individual dispatch() calls for inference.
    ///
    /// # Safety
    /// All resources must remain alive until completion.
    #[inline(always)]
    pub unsafe fn dispatch_batch<F>(&self, encode: F)
    where
        F: FnOnce(&BatchEncoder),
    {
        let cmd = msg0_retained(self.queue, self.s_cb);
        let enc = msg0_retained(cmd, self.s_ce);

        let batch = BatchEncoder {
            enc,
            imp_sp: self.imp_sp,
            imp_sb: self.imp_sb,
            imp_by: self.imp_by,
            imp_dt: self.imp_dt,
            imp_dg: self.imp_dg,
            s_sp: self.s_sp,
            s_sb: self.s_sb,
            s_by: self.s_by,
            s_dt: self.s_dt,
            s_dg: self.s_dg,
        };

        encode(&batch);

        (self.imp_ee)(enc, self.s_ee);
        (self.imp_cm)(cmd, self.s_cm);
        (self.imp_wt)(cmd, self.s_wt);

        objc_release(enc);
        objc_release(cmd);
    }

    /// Like `dispatch_batch` but without autorelease pool management.
    /// Caller MUST be inside an `autorelease_pool` scope.
    /// Use this for multi-batch inference loops where the caller manages
    /// a single pool for the entire pass.
    ///
    /// # Safety
    /// Must be inside `autorelease_pool`. All resources must remain alive.
    #[inline(always)]
    pub unsafe fn dispatch_batch_raw<F>(&self, encode: F)
    where
        F: FnOnce(&BatchEncoder),
    {
        let cmd = (self.imp_cb)(self.queue, self.s_cb);
        debug_assert!(!cmd.is_null(), "command buffer creation returned null");
        let enc = (self.imp_ce)(cmd, self.s_ce);
        debug_assert!(!enc.is_null(), "compute encoder creation returned null");

        let batch = BatchEncoder {
            enc,
            imp_sp: self.imp_sp,
            imp_sb: self.imp_sb,
            imp_by: self.imp_by,
            imp_dt: self.imp_dt,
            imp_dg: self.imp_dg,
            s_sp: self.s_sp,
            s_sb: self.s_sb,
            s_by: self.s_by,
            s_dt: self.s_dt,
            s_dg: self.s_dg,
        };

        encode(&batch);

        (self.imp_ee)(enc, self.s_ee);
        (self.imp_cm)(cmd, self.s_cm);
        (self.imp_wt)(cmd, self.s_wt);
    }

    /// Pipelined dispatch: encode + commit, return handle for deferred wait.
    /// Overlap GPU execution of batch N with CPU encoding of batch N+1.
    ///
    /// Usage:
    /// ```ignore
    /// let mut prev = None;
    /// for pass in passes {
    ///     let handle = disp.dispatch_batch_async(|batch| { ... });
    ///     if let Some(h) = prev { h.wait(); }
    ///     prev = Some(handle);
    /// }
    /// if let Some(h) = prev { h.wait(); }
    /// ```
    ///
    /// # Safety
    /// All resources must remain alive until `GpuFuture::wait()` is called.
    #[inline(always)]
    pub unsafe fn dispatch_batch_async<F>(&self, encode: F) -> GpuFuture
    where
        F: FnOnce(&BatchEncoder),
    {
        let cmd = msg0_retained(self.queue, self.s_cb);
        let enc = msg0_retained(cmd, self.s_ce);

        let batch = BatchEncoder {
            enc,
            imp_sp: self.imp_sp,
            imp_sb: self.imp_sb,
            imp_by: self.imp_by,
            imp_dt: self.imp_dt,
            imp_dg: self.imp_dg,
            s_sp: self.s_sp,
            s_sb: self.s_sb,
            s_by: self.s_by,
            s_dt: self.s_dt,
            s_dg: self.s_dg,
        };

        encode(&batch);

        (self.imp_ee)(enc, self.s_ee);
        (self.imp_cm)(cmd, self.s_cm);
        // Do NOT wait — return handle for deferred wait

        objc_release(enc);
        GpuFuture {
            cmd,
            s_wt: self.s_wt,
        }
    }
}

impl Drop for ComputeDispatcher {
    fn drop(&mut self) {
        unsafe { objc_release(self.queue) };
    }
}

/// Handle for a committed but not yet completed GPU command buffer.
/// Call `wait()` to block until execution finishes.
pub struct GpuFuture {
    cmd: ObjcId,
    s_wt: ObjcSel,
}

impl GpuFuture {
    /// Block until the GPU finishes executing this command buffer.
    #[inline(always)]
    pub fn wait(self) {
        unsafe {
            msg0_void(self.cmd, self.s_wt);
            objc_release(self.cmd);
        }
        std::mem::forget(self); // prevent Drop from double-releasing
    }
}

impl Drop for GpuFuture {
    fn drop(&mut self) {
        // If not waited, wait + release to prevent leak
        unsafe {
            msg0_void(self.cmd, self.s_wt);
            objc_release(self.cmd);
        }
    }
}

/// A batch encoder for encoding multiple dispatches into one command buffer.
pub struct BatchEncoder {
    enc: ObjcId,
    imp_sp: ImpObj,
    imp_sb: ImpBuf,
    imp_by: ImpBytes,
    imp_dt: ImpDisp,
    imp_dg: ImpDisp,
    s_sp: ObjcSel,
    s_sb: ObjcSel,
    s_by: ObjcSel,
    s_dt: ObjcSel,
    s_dg: ObjcSel,
}

impl BatchEncoder {
    #[inline(always)]
    pub fn set_pipeline(&self, pipeline: &MtlComputePipeline) {
        unsafe { (self.imp_sp)(self.enc, self.s_sp, pipeline.as_raw()) };
    }

    #[inline(always)]
    pub fn set_buffer(&self, buffer: &MtlBuffer, offset: usize, index: usize) {
        unsafe { (self.imp_sb)(self.enc, self.s_sb, buffer.as_raw(), offset, index) };
    }

    #[inline(always)]
    pub fn set_bytes(&self, data: &[u8], index: usize) {
        unsafe {
            (self.imp_by)(
                self.enc,
                self.s_by,
                data.as_ptr() as *const c_void,
                data.len(),
                index,
            )
        };
    }

    #[inline(always)]
    pub fn dispatch_threads(&self, grid: (usize, usize, usize), group: (usize, usize, usize)) {
        let g = MTLSize {
            width: grid.0,
            height: grid.1,
            depth: grid.2,
        };
        let t = MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        };
        unsafe { (self.imp_dt)(self.enc, self.s_dt, g, t) };
    }

    #[inline(always)]
    pub fn dispatch_threadgroups(
        &self,
        groups: (usize, usize, usize),
        threads: (usize, usize, usize),
    ) {
        let g = MTLSize {
            width: groups.0,
            height: groups.1,
            depth: groups.2,
        };
        let t = MTLSize {
            width: threads.0,
            height: threads.1,
            depth: threads.2,
        };
        unsafe { (self.imp_dg)(self.enc, self.s_dg, g, t) };
    }
}
