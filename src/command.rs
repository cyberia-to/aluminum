//! MtlCommandQueue + MtlCommandBuffer

use crate::encoder::{MtlBlitEncoder, MtlComputeEncoder};
use crate::ffi::*;
use crate::MetalError;

#[cold]
#[inline(never)]
fn err_cmd_buffer() -> MetalError {
    MetalError::CommandBufferError("creation failed".into())
}

#[cold]
#[inline(never)]
fn err_encoder() -> MetalError {
    MetalError::EncoderCreationFailed
}

/// A command queue for submitting work to the GPU. Wraps `id<MTLCommandQueue>`.
pub struct MtlCommandQueue {
    raw: ObjcId,
}

impl MtlCommandQueue {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlCommandQueue { raw }
    }

    /// Create a new command buffer (retained).
    /// Uses ARC fast-retain to skip autorelease+retain round-trip.
    #[inline(always)]
    pub fn command_buffer(&self) -> Result<MtlCommandBuffer, MetalError> {
        let raw = unsafe { msg0_retained(self.raw, SEL_commandBuffer()) };
        if raw.is_null() {
            return Err(err_cmd_buffer());
        }
        Ok(MtlCommandBuffer { raw, owned: true })
    }

    /// Create a command buffer without retain/release overhead.
    ///
    /// # Safety
    /// The returned buffer is autoreleased. Caller must ensure an autorelease pool
    /// is active and that the buffer is not used after the pool is drained.
    /// Use `autorelease_pool()` to create a pool scope.
    #[inline]
    pub unsafe fn command_buffer_unretained(&self) -> Result<MtlCommandBuffer, MetalError> {
        let raw = msg0(self.raw, SEL_commandBuffer());
        if raw.is_null() {
            return Err(err_cmd_buffer());
        }
        Ok(MtlCommandBuffer { raw, owned: false })
    }

    /// Create a command buffer that does NOT retain referenced resources.
    /// Faster than `command_buffer()` — Metal skips retain/release on all
    /// buffers, textures, and pipeline states used in the command buffer.
    ///
    /// # Safety
    /// Caller must ensure all resources referenced by encoded commands
    /// remain alive until the command buffer completes.
    #[inline(always)]
    pub unsafe fn command_buffer_fast(&self) -> Result<MtlCommandBuffer, MetalError> {
        let raw = msg0(self.raw, SEL_commandBufferWithUnretainedReferences());
        if raw.is_null() {
            return Err(err_cmd_buffer());
        }
        objc_retain(raw);
        Ok(MtlCommandBuffer { raw, owned: true })
    }

    /// Zero-overhead command buffer: unretained references, no null check, no Result.
    /// Uses `commandBufferWithUnretainedReferences` — Metal skips internal
    /// retain/release on all encoded resources.
    ///
    /// # Safety
    /// All resources referenced by encoded commands must remain alive until completion.
    #[inline(always)]
    pub unsafe fn command_buffer_unchecked(&self) -> MtlCommandBuffer {
        let raw = msg0_retained(self.raw, SEL_commandBufferWithUnretainedReferences());
        MtlCommandBuffer { raw, owned: true }
    }

    /// Fastest possible command buffer: no retain, no release, no null check.
    /// Object is autoreleased — caller MUST be inside an `autorelease_pool`.
    /// Uses `commandBufferWithUnretainedReferences` for Metal-side savings too.
    ///
    /// # Safety
    /// Must be inside `autorelease_pool`. Object invalid after pool drains.
    /// All encoded resources must outlive the command buffer.
    #[inline(always)]
    pub unsafe fn command_buffer_autoreleased(&self) -> MtlCommandBuffer {
        let raw = msg0(self.raw, SEL_commandBufferWithUnretainedReferences());
        MtlCommandBuffer { raw, owned: false }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlCommandQueue {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A command buffer for encoding GPU commands. Wraps `id<MTLCommandBuffer>`.
pub struct MtlCommandBuffer {
    raw: ObjcId,
    owned: bool,
}

impl MtlCommandBuffer {
    /// Create a compute command encoder (retained).
    /// Uses ARC fast-retain to skip autorelease+retain round-trip.
    #[inline(always)]
    pub fn compute_encoder(&self) -> Result<MtlComputeEncoder, MetalError> {
        let raw = unsafe { msg0_retained(self.raw, SEL_computeCommandEncoder()) };
        if raw.is_null() {
            return Err(err_encoder());
        }
        Ok(MtlComputeEncoder::from_raw(raw, true))
    }

    /// Create a compute encoder without retain/release overhead.
    ///
    /// # Safety
    /// Same as `command_buffer_unretained`.
    #[inline]
    pub unsafe fn compute_encoder_unretained(&self) -> Result<MtlComputeEncoder, MetalError> {
        let raw = msg0(self.raw, SEL_computeCommandEncoder());
        if raw.is_null() {
            return Err(err_encoder());
        }
        Ok(MtlComputeEncoder::from_raw(raw, false))
    }

    /// Zero-overhead compute encoder: no null check, no Result.
    ///
    /// # Safety
    /// Command buffer must be in a valid state for encoding.
    #[inline(always)]
    pub unsafe fn compute_encoder_unchecked(&self) -> MtlComputeEncoder {
        let raw = msg0_retained(self.raw, SEL_computeCommandEncoder());
        MtlComputeEncoder::from_raw(raw, true)
    }

    /// Fastest encoder: no retain, no release, no null check.
    /// Autoreleased — valid only within current autorelease pool.
    ///
    /// # Safety
    /// Must be inside `autorelease_pool`. Command buffer must be valid.
    #[inline(always)]
    pub unsafe fn compute_encoder_autoreleased(&self) -> MtlComputeEncoder {
        let raw = msg0(self.raw, SEL_computeCommandEncoder());
        MtlComputeEncoder::from_raw(raw, false)
    }

    /// Create a blit command encoder.
    #[inline]
    pub fn blit_encoder(&self) -> Result<MtlBlitEncoder, MetalError> {
        let raw = unsafe { msg0(self.raw, SEL_blitCommandEncoder()) };
        if raw.is_null() {
            return Err(err_encoder());
        }
        unsafe { retain(raw) };
        Ok(MtlBlitEncoder::from_raw(raw))
    }

    /// Commit the command buffer for execution.
    #[inline(always)]
    pub fn commit(&self) {
        unsafe { msg0_void(self.raw, SEL_commit()) };
    }

    /// Block until the command buffer completes.
    #[inline(always)]
    pub fn wait_until_completed(&self) {
        unsafe { msg0_void(self.raw, SEL_waitUntilCompleted()) };
    }

    /// Command buffer status.
    pub fn status(&self) -> u64 {
        unsafe { msg0_u64(self.raw, SEL_status()) }
    }

    /// Get error description if the command buffer failed.
    pub fn error(&self) -> Option<String> {
        let err = unsafe { msg0(self.raw, SEL_error()) };
        nserror_string(err)
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlCommandBuffer {
    fn drop(&mut self) {
        if self.owned {
            unsafe { release_nonnull(self.raw) };
        }
    }
}
