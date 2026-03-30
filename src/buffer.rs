//! MtlBuffer: zero-copy shared GPU memory (replaces IOSurface for Metal)

use crate::ffi::*;
use std::ffi::c_void;

/// A Metal buffer with shared CPU/GPU memory. Wraps `id<MTLBuffer>`.
///
/// Created via `MtlDevice::new_buffer()`. Uses `MTLResourceStorageModeShared`
/// so CPU and GPU share the same physical memory — no copies, no lock/unlock.
pub struct MtlBuffer {
    raw: ObjcId,
    size: usize,
    /// Cached contents pointer — valid for buffer lifetime with shared storage mode.
    ptr: *mut c_void,
}

impl MtlBuffer {
    pub(crate) fn from_raw(raw: ObjcId, size: usize) -> Self {
        let ptr = unsafe { msg0_ptr(raw, SEL_contents()) };
        MtlBuffer { raw, size, ptr }
    }

    /// Create from raw without querying contents (for Private storage mode).
    pub(crate) fn from_raw_private(raw: ObjcId, size: usize) -> Self {
        MtlBuffer {
            raw,
            size,
            ptr: std::ptr::null_mut(),
        }
    }

    /// Whether this buffer has CPU-accessible memory (Shared mode).
    #[inline(always)]
    pub fn is_shared(&self) -> bool {
        !self.ptr.is_null()
    }

    /// Raw pointer to buffer contents. Cached — no ObjC call after first access.
    #[inline(always)]
    pub fn contents(&self) -> *mut c_void {
        self.ptr
    }

    /// Read access to buffer data via closure.
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer (not CPU-accessible).
    #[inline]
    #[track_caller]
    pub fn with_data<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        assert!(self.is_shared(), "with_data called on private buffer");
        let slice = unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.size) };
        f(slice)
    }

    /// Write access to buffer data via closure.
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer (not CPU-accessible).
    #[inline]
    #[track_caller]
    pub fn with_data_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        assert!(self.is_shared(), "with_data_mut called on private buffer");
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.size) };
        f(slice)
    }

    /// Read access as f32 slice.
    ///
    /// Length = `size / 4` (trailing bytes not divisible by 4 are ignored).
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer or if contents pointer is not 4-byte aligned.
    #[inline]
    #[track_caller]
    pub fn with_f32<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[f32]) -> R,
    {
        assert!(self.is_shared(), "with_f32 called on private buffer");
        let ptr = self.ptr as *const f32;
        assert!(ptr.is_aligned(), "buffer contents not 4-byte aligned");
        let len = self.size / 4;
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        f(slice)
    }

    /// Write access as f32 slice.
    ///
    /// Length = `size / 4` (trailing bytes not divisible by 4 are ignored).
    ///
    /// # Panics
    /// Panics if called on a private-mode buffer or if contents pointer is not 4-byte aligned.
    #[inline]
    #[track_caller]
    pub fn with_f32_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [f32]) -> R,
    {
        assert!(self.is_shared(), "with_f32_mut called on private buffer");
        let ptr = self.ptr as *mut f32;
        assert!(ptr.is_aligned(), "buffer contents not 4-byte aligned");
        let len = self.size / 4;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        f(slice)
    }

    /// Allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlBuffer {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
