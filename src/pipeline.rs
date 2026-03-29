//! MtlComputePipeline + MtlRenderPipeline

use crate::ffi::*;

/// A compiled compute pipeline state. Wraps `id<MTLComputePipelineState>`.
pub struct MtlComputePipeline {
    raw: ObjcId,
}

impl MtlComputePipeline {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlComputePipeline { raw }
    }

    /// Maximum total threads per threadgroup for this pipeline.
    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_maxTotalThreadsPerThreadgroup()) }
    }

    /// Thread execution width (SIMD width).
    pub fn thread_execution_width(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_threadExecutionWidth()) }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlComputePipeline {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A compiled render pipeline state. Wraps `id<MTLRenderPipelineState>`.
pub struct MtlRenderPipeline {
    raw: ObjcId,
}

impl MtlRenderPipeline {
    #[allow(dead_code)]
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlRenderPipeline { raw }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlRenderPipeline {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
