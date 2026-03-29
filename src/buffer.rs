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
    #[inline]
    pub fn with_data<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        let ptr = self.contents() as *const u8;
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.size) };
        f(slice)
    }

    /// Write access to buffer data via closure.
    #[inline]
    pub fn with_data_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let ptr = self.contents() as *mut u8;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.size) };
        f(slice)
    }

    /// Read access as f32 slice.
    #[inline]
    pub fn with_f32<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&[f32]) -> R,
    {
        let ptr = self.contents() as *const f32;
        let len = self.size / 4;
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        f(slice)
    }

    /// Write access as f32 slice.
    #[inline]
    pub fn with_f32_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut [f32]) -> R,
    {
        let ptr = self.contents() as *mut f32;
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

// ── fp16 conversion (same as rANE surface.rs) ──

/// Decode fp16 to f32 — NEON fcvt on aarch64
#[inline(always)]
pub fn fp16_to_f32(v: u16) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        let result: u32;
        unsafe {
            std::arch::asm!(
                "fmov h0, {src:w}",
                "fcvt s0, h0",
                "fmov {dst:w}, s0",
                src = in(reg) v as u32,
                dst = out(reg) result,
                out("v0") _,
            );
        }
        f32::from_bits(result)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        fp16_to_f32_soft(v)
    }
}

/// Encode f32 to fp16 — NEON fcvt on aarch64
#[inline(always)]
pub fn f32_to_fp16(v: f32) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        let result: u32;
        unsafe {
            std::arch::asm!(
                "fmov s0, {src:w}",
                "fcvt h0, s0",
                "fmov {dst:w}, s0",
                src = in(reg) v.to_bits(),
                dst = out(reg) result,
                out("v0") _,
            );
        }
        result as u16
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32_to_fp16_soft(v)
    }
}

/// Bulk convert fp16 → f32 using NEON (32 elements per iteration, 4x unrolled)
pub fn cvt_f16_f32(dst: &mut [f32], src: &[u16]) {
    let n = dst.len().min(src.len());
    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        // 4x unrolled: 32 fp16 → 32 f32 per iteration
        while i + 32 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    "ldp q0, q1, [{s}]",       // 16 fp16 (32 bytes)
                    "ldp q2, q3, [{s}, #32]",   // 16 fp16 (32 bytes)
                    "fcvtl v4.4s, v0.4h",
                    "fcvtl2 v5.4s, v0.8h",
                    "fcvtl v6.4s, v1.4h",
                    "fcvtl2 v7.4s, v1.8h",
                    "fcvtl v16.4s, v2.4h",
                    "fcvtl2 v17.4s, v2.8h",
                    "fcvtl v18.4s, v3.4h",
                    "fcvtl2 v19.4s, v3.8h",
                    "stp q4, q5, [{d}]",        // 8 f32
                    "stp q6, q7, [{d}, #32]",   // 8 f32
                    "stp q16, q17, [{d}, #64]", // 8 f32
                    "stp q18, q19, [{d}, #96]", // 8 f32
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                    out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                );
            }
            i += 32;
        }
        // Tail: 8 at a time
        while i + 8 <= n {
            unsafe {
                std::arch::asm!(
                    "ldr q0, [{src}]",
                    "fcvtl v1.4s, v0.4h",
                    "fcvtl2 v2.4s, v0.8h",
                    "stp q1, q2, [{dst}]",
                    src = in(reg) src.as_ptr().add(i),
                    dst = in(reg) dst.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _, out("v2") _,
                );
            }
            i += 8;
        }
        for j in i..n {
            dst[j] = fp16_to_f32(src[j]);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = fp16_to_f32_soft(src[i]);
        }
    }
}

/// Bulk convert f32 → fp16 using NEON (32 elements per iteration, 4x unrolled)
pub fn cvt_f32_f16(dst: &mut [u16], src: &[f32]) {
    let n = dst.len().min(src.len());
    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        // 4x unrolled: 32 f32 → 32 fp16 per iteration
        while i + 32 <= n {
            unsafe {
                let s = src.as_ptr().add(i);
                let d = dst.as_mut_ptr().add(i);
                std::arch::asm!(
                    "ldp q0, q1, [{s}]",        // 8 f32
                    "ldp q2, q3, [{s}, #32]",   // 8 f32
                    "ldp q4, q5, [{s}, #64]",   // 8 f32
                    "ldp q6, q7, [{s}, #96]",   // 8 f32
                    "fcvtn v0.4h, v0.4s",
                    "fcvtn2 v0.8h, v1.4s",
                    "fcvtn v2.4h, v2.4s",
                    "fcvtn2 v2.8h, v3.4s",
                    "fcvtn v4.4h, v4.4s",
                    "fcvtn2 v4.8h, v5.4s",
                    "fcvtn v6.4h, v6.4s",
                    "fcvtn2 v6.8h, v7.4s",
                    "stp q0, q2, [{d}]",        // 16 fp16
                    "stp q4, q6, [{d}, #32]",   // 16 fp16
                    s = in(reg) s,
                    d = in(reg) d,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                );
            }
            i += 32;
        }
        // Tail: 8 at a time
        while i + 8 <= n {
            unsafe {
                std::arch::asm!(
                    "ldp q0, q1, [{src}]",
                    "fcvtn v2.4h, v0.4s",
                    "fcvtn2 v2.8h, v1.4s",
                    "str q2, [{dst}]",
                    src = in(reg) src.as_ptr().add(i),
                    dst = in(reg) dst.as_mut_ptr().add(i),
                    out("v0") _, out("v1") _, out("v2") _,
                );
            }
            i += 8;
        }
        for j in i..n {
            dst[j] = f32_to_fp16(src[j]);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            dst[i] = f32_to_fp16_soft(src[i]);
        }
    }
}

#[allow(dead_code)]
fn fp16_to_f32_soft(v: u16) -> f32 {
    let sign = (v >> 15) & 1;
    let exp = (v >> 10) & 0x1F;
    let frac = v & 0x3FF;
    if exp == 0 {
        ((-1.0f32).powi(sign as i32)) * (frac as f32 / 1024.0) * 2.0f32.powi(-14)
    } else if exp == 31 {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        ((-1.0f32).powi(sign as i32)) * (1.0 + frac as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    }
}

#[allow(dead_code)]
fn f32_to_fp16_soft(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let frac = bits & 0x7FFFFF;
    if exp > 15 {
        ((sign << 15) | 0x7C00) as u16
    } else if exp < -14 {
        if exp < -24 {
            (sign << 15) as u16
        } else {
            ((sign << 15) | ((0x800000 | frac) >> (-1 - exp + 13))) as u16
        }
    } else {
        ((sign << 15) | (((exp + 15) as u32) << 10) | (frac >> 13)) as u16
    }
}
