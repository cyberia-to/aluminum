//! Synchronization primitives: MtlFence, MtlEvent, MtlSharedEvent

use crate::ffi::*;

/// A fence for tracking GPU work within a command buffer. Wraps `id<MTLFence>`.
pub struct MtlFence {
    raw: ObjcId,
}

impl MtlFence {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlFence { raw }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlFence {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A GPU event for synchronizing command buffer execution. Wraps `id<MTLEvent>`.
pub struct MtlEvent {
    raw: ObjcId,
}

impl MtlEvent {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlEvent { raw }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlEvent {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A shared event for CPU/GPU synchronization. Wraps `id<MTLSharedEvent>`.
pub struct MtlSharedEvent {
    raw: ObjcId,
}

impl MtlSharedEvent {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlSharedEvent { raw }
    }

    /// Get the current signaled value.
    pub fn signaled_value(&self) -> u64 {
        unsafe { msg0_u64(self.raw, SEL_signaledValue()) }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlSharedEvent {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
