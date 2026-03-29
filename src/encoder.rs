//! MtlComputeEncoder + MtlBlitEncoder

use crate::buffer::MtlBuffer;
use crate::ffi::*;
use crate::pipeline::MtlComputePipeline;
use std::ffi::c_void;

/// A compute command encoder. Wraps `id<MTLComputeCommandEncoder>`.
pub struct MtlComputeEncoder {
    raw: ObjcId,
    owned: bool,
}

impl MtlComputeEncoder {
    pub(crate) fn from_raw(raw: ObjcId, owned: bool) -> Self {
        MtlComputeEncoder { raw, owned }
    }

    /// Set the compute pipeline state.
    #[inline(always)]
    pub fn set_pipeline(&self, pipeline: &MtlComputePipeline) {
        unsafe { msg1_void(self.raw, SEL_setComputePipelineState(), pipeline.as_raw()) };
    }

    /// Set a buffer at an index.
    #[inline(always)]
    pub fn set_buffer(&self, buffer: &MtlBuffer, offset: usize, index: usize) {
        unsafe {
            msg3_void(
                self.raw,
                SEL_setBuffer_offset_atIndex(),
                buffer.as_raw(),
                offset,
                index,
            )
        };
    }

    /// Set inline bytes at an index.
    #[inline(always)]
    pub fn set_bytes(&self, data: &[u8], index: usize) {
        unsafe {
            msg_bytes_void(
                self.raw,
                SEL_setBytes_length_atIndex(),
                data.as_ptr() as *const c_void,
                data.len(),
                index,
            )
        };
    }

    /// Dispatch threads with automatic threadgroup sizing.
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
        unsafe { msg_dispatch_void(self.raw, SEL_dispatchThreads(), g, t) };
    }

    /// Dispatch threadgroups with explicit group count.
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
        unsafe { msg_dispatch_void(self.raw, SEL_dispatchThreadgroups(), g, t) };
    }

    /// End encoding. Must be called before committing the command buffer.
    #[inline(always)]
    pub fn end_encoding(&self) {
        unsafe { msg0_void(self.raw, SEL_endEncoding()) };
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlComputeEncoder {
    fn drop(&mut self) {
        if self.owned {
            unsafe { release_nonnull(self.raw) };
        }
    }
}

/// A blit command encoder. Wraps `id<MTLBlitCommandEncoder>`.
pub struct MtlBlitEncoder {
    raw: ObjcId,
}

impl MtlBlitEncoder {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlBlitEncoder { raw }
    }

    /// Copy from one buffer to another.
    pub fn copy_buffer(
        &self,
        src: &MtlBuffer,
        src_offset: usize,
        dst: &MtlBuffer,
        dst_offset: usize,
        size: usize,
    ) {
        unsafe {
            type F = unsafe extern "C" fn(
                ObjcId,
                ObjcSel,
                ObjcId,
                NSUInteger,
                ObjcId,
                NSUInteger,
                NSUInteger,
            );
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_copyFromBuffer(),
                src.as_raw(),
                src_offset,
                dst.as_raw(),
                dst_offset,
                size,
            );
        }
    }

    /// End encoding.
    #[inline(always)]
    pub fn end_encoding(&self) {
        unsafe { msg0_void(self.raw, SEL_endEncoding()) };
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlBlitEncoder {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
