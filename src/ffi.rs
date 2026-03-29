//! Raw FFI bindings to Apple frameworks: Metal, CoreFoundation, libobjc

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::missing_safety_doc
)]

use std::ffi::{c_char, c_void, CStr};

// ── Type aliases ──

pub type ObjcClass = *const c_void;
pub type ObjcSel = *const c_void;
pub type ObjcId = *mut c_void;
pub type CFTypeRef = *const c_void;
pub type CFStringRef = *const c_void;
pub type CFMutableDictionaryRef = *mut c_void;
pub type NSUInteger = usize;

// ── MTLSize (used for dispatch) ──

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLSize {
    pub width: NSUInteger,
    pub height: NSUInteger,
    pub depth: NSUInteger,
}

// ── MTLRegion ──

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLOrigin {
    pub x: NSUInteger,
    pub y: NSUInteger,
    pub z: NSUInteger,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLRegion {
    pub origin: MTLOrigin,
    pub size: MTLSize,
}

// ── Metal resource options ──

pub const MTLResourceStorageModeShared: NSUInteger = 0;
pub const MTLResourceStorageModeManaged: NSUInteger = 0x10;
pub const MTLResourceStorageModePrivate: NSUInteger = 0x20;

// ── Metal pixel formats (common) ──

pub const MTLPixelFormatRGBA8Unorm: NSUInteger = 70;
pub const MTLPixelFormatBGRA8Unorm: NSUInteger = 80;
pub const MTLPixelFormatR32Float: NSUInteger = 55;
pub const MTLPixelFormatRGBA32Float: NSUInteger = 125;

// ── CoreFoundation constants ──

pub const kCFStringEncodingUTF8: u32 = 0x08000100;

// ── Metal.framework ──

#[link(name = "Metal", kind = "framework")]
extern "C" {
    pub fn MTLCreateSystemDefaultDevice() -> ObjcId;
    pub fn MTLCopyAllDevices() -> ObjcId;
}

// ── CoreFoundation ──

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    pub fn CFStringCreateWithCString(
        alloc: *const c_void,
        cStr: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    pub fn CFStringCreateWithBytes(
        alloc: *const c_void,
        bytes: *const u8,
        numBytes: isize,
        encoding: u32,
        isExternalRepresentation: bool,
    ) -> CFStringRef;
    pub fn CFRelease(cf: CFTypeRef);
    pub fn CFRetain(cf: CFTypeRef) -> CFTypeRef;
}

// ── ObjC runtime ──

extern "C" {
    pub fn objc_getClass(name: *const c_char) -> ObjcClass;
    pub fn sel_registerName(name: *const c_char) -> ObjcSel;
    pub fn objc_msgSend() -> ObjcId;
    pub fn objc_retain(obj: ObjcId) -> ObjcId;
    pub fn objc_release(obj: ObjcId);
    pub fn objc_autoreleasePoolPush() -> *mut c_void;
    pub fn objc_autoreleasePoolPop(pool: *mut c_void);
    pub fn object_getClass(obj: ObjcId) -> ObjcClass;
    pub fn class_getMethodImplementation(cls: ObjcClass, sel: ObjcSel) -> *const c_void;
    /// ARC optimization: if called immediately after a msg_send that returns
    /// an autoreleased object, the runtime skips the autorelease+retain
    /// round-trip entirely. Must be called with no intervening code.
    pub fn objc_retainAutoreleasedReturnValue(obj: ObjcId) -> ObjcId;
}

// ── Cached selectors ──
// sel_registerName is idempotent: same C string → same pointer.
// We cache the result via AtomicPtr with Relaxed ordering — on ARM64 this
// compiles to a plain `ldr` (no memory barrier). The race on first init is
// benign: sel_registerName always returns the same value for the same string.

macro_rules! cached_sel {
    ($name:ident, $lit:expr) => {
        #[inline(always)]
        pub fn $name() -> ObjcSel {
            use std::sync::atomic::{AtomicPtr, Ordering};
            static CACHE: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
            let p = CACHE.load(Ordering::Relaxed);
            if !p.is_null() {
                return p as ObjcSel;
            }
            let s = unsafe { sel_registerName($lit.as_ptr()) };
            CACHE.store(s as *mut (), Ordering::Relaxed);
            s
        }
    };
}

// Hot path selectors — called on every command buffer dispatch
cached_sel!(SEL_retain, c"retain");
cached_sel!(SEL_release, c"release");
cached_sel!(SEL_commandBuffer, c"commandBuffer");
cached_sel!(SEL_computeCommandEncoder, c"computeCommandEncoder");
cached_sel!(SEL_setComputePipelineState, c"setComputePipelineState:");
cached_sel!(SEL_setBuffer_offset_atIndex, c"setBuffer:offset:atIndex:");
cached_sel!(SEL_setBytes_length_atIndex, c"setBytes:length:atIndex:");
cached_sel!(
    SEL_dispatchThreads,
    c"dispatchThreads:threadsPerThreadgroup:"
);
cached_sel!(
    SEL_dispatchThreadgroups,
    c"dispatchThreadgroups:threadsPerThreadgroup:"
);
cached_sel!(SEL_endEncoding, c"endEncoding");
cached_sel!(SEL_commit, c"commit");
cached_sel!(SEL_waitUntilCompleted, c"waitUntilCompleted");
cached_sel!(SEL_contents, c"contents");
cached_sel!(SEL_status, c"status");
cached_sel!(SEL_error, c"error");

// Fast command buffer (unretained references)
cached_sel!(
    SEL_commandBufferWithUnretainedReferences,
    c"commandBufferWithUnretainedReferences"
);

// Device selectors
cached_sel!(SEL_name, c"name");
cached_sel!(SEL_newCommandQueue, c"newCommandQueue");
cached_sel!(
    SEL_newBufferWithLength_options,
    c"newBufferWithLength:options:"
);
cached_sel!(
    SEL_newBufferWithBytes_length_options,
    c"newBufferWithBytes:length:options:"
);
cached_sel!(
    SEL_newLibraryWithSource_options_error,
    c"newLibraryWithSource:options:error:"
);
cached_sel!(
    SEL_newComputePipelineStateWithFunction_error,
    c"newComputePipelineStateWithFunction:error:"
);
cached_sel!(SEL_hasUnifiedMemory, c"hasUnifiedMemory");
cached_sel!(SEL_maxBufferLength, c"maxBufferLength");
cached_sel!(SEL_maxThreadsPerThreadgroup, c"maxThreadsPerThreadgroup");
cached_sel!(
    SEL_recommendedMaxWorkingSetSize,
    c"recommendedMaxWorkingSetSize"
);
cached_sel!(SEL_newTextureWithDescriptor, c"newTextureWithDescriptor:");
cached_sel!(SEL_newFence, c"newFence");
cached_sel!(SEL_newEvent, c"newEvent");
cached_sel!(SEL_newSharedEvent, c"newSharedEvent");

// Library/function selectors
cached_sel!(SEL_newFunctionWithName, c"newFunctionWithName:");
cached_sel!(SEL_functionNames, c"functionNames");

// Pipeline selectors
cached_sel!(
    SEL_maxTotalThreadsPerThreadgroup,
    c"maxTotalThreadsPerThreadgroup"
);
cached_sel!(SEL_threadExecutionWidth, c"threadExecutionWidth");

// Collection selectors
cached_sel!(SEL_count, c"count");
cached_sel!(SEL_objectAtIndex, c"objectAtIndex:");

// Blit encoder
cached_sel!(SEL_blitCommandEncoder, c"blitCommandEncoder");
cached_sel!(
    SEL_copyFromBuffer,
    c"copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
);

// Texture
cached_sel!(SEL_width, c"width");
cached_sel!(SEL_height, c"height");
cached_sel!(SEL_depth, c"depth");
cached_sel!(SEL_pixelFormat, c"pixelFormat");
cached_sel!(
    SEL_replaceRegion,
    c"replaceRegion:mipmapLevel:withBytes:bytesPerRow:"
);
cached_sel!(
    SEL_getBytes,
    c"getBytes:bytesPerRow:fromRegion:mipmapLevel:"
);

// Sync
cached_sel!(SEL_signaledValue, c"signaledValue");

// NSString / NSError
cached_sel!(SEL_localizedDescription, c"localizedDescription");
cached_sel!(SEL_UTF8String, c"UTF8String");
cached_sel!(SEL_stringWithUTF8String, c"stringWithUTF8String:");

// ── Cached classes ──

macro_rules! cached_cls {
    ($name:ident, $lit:expr) => {
        #[inline(always)]
        pub fn $name() -> ObjcClass {
            use std::sync::atomic::{AtomicPtr, Ordering};
            static CACHE: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
            let p = CACHE.load(Ordering::Relaxed);
            if !p.is_null() {
                return p as ObjcClass;
            }
            let s = unsafe { objc_getClass($lit.as_ptr()) };
            CACHE.store(s as *mut (), Ordering::Relaxed);
            s
        }
    };
}

cached_cls!(CLS_NSString, c"NSString");
cached_cls!(CLS_NSArray, c"NSArray");
cached_cls!(CLS_NSDictionary, c"NSDictionary");

// ── Helpers (fallback for dynamic/rare selectors) ──

pub fn sel(name: &str) -> ObjcSel {
    let c = std::ffi::CString::new(name).unwrap();
    unsafe { sel_registerName(c.as_ptr()) }
}

pub fn cls(name: &str) -> ObjcClass {
    let c = std::ffi::CString::new(name).unwrap();
    unsafe { objc_getClass(c.as_ptr()) }
}

// ── objc_msgSend typed trampolines ──
// The transmute is done once per call-site type signature.
// Selectors are passed as pre-cached ObjcSel, not &str.

/// Send a message with 0 args, returning ObjcId.
#[inline(always)]
pub unsafe fn msg0(target: ObjcId, sel: ObjcSel) -> ObjcId {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send msg + immediately retain the autoreleased return value.
/// Uses ARC fast-path: runtime skips autorelease+retain round-trip
/// when retain is called immediately after msg_send.
#[inline(always)]
pub unsafe fn msg0_retained(target: ObjcId, sel: ObjcSel) -> ObjcId {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    let obj = f(target, sel);
    objc_retainAutoreleasedReturnValue(obj)
}

/// Send a message with 0 args, returning nothing.
#[inline(always)]
pub unsafe fn msg0_void(target: ObjcId, sel: ObjcSel) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel);
}

/// Send a message with 0 args, returning bool.
#[inline(always)]
pub unsafe fn msg0_bool(target: ObjcId, sel: ObjcSel) -> bool {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> bool;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning usize.
#[inline(always)]
pub unsafe fn msg0_usize(target: ObjcId, sel: ObjcSel) -> NSUInteger {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> NSUInteger;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning u64.
#[inline(always)]
pub unsafe fn msg0_u64(target: ObjcId, sel: ObjcSel) -> u64 {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> u64;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning *mut c_void.
#[inline(always)]
pub unsafe fn msg0_ptr(target: ObjcId, sel: ObjcSel) -> *mut c_void {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> *mut c_void;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 0 args, returning MTLSize.
#[inline(always)]
pub unsafe fn msg0_mtlsize(target: ObjcId, sel: ObjcSel) -> MTLSize {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> MTLSize;
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel)
}

/// Send a message with 1 arg.
#[inline(always)]
pub unsafe fn msg1<A>(target: ObjcId, sel: ObjcSel, a: A) -> ObjcId {
    type F<A> = unsafe extern "C" fn(ObjcId, ObjcSel, A) -> ObjcId;
    let f: F<A> = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a)
}

/// Send a message with 2 args.
#[inline(always)]
pub unsafe fn msg2<A, B>(target: ObjcId, sel: ObjcSel, a: A, b: B) -> ObjcId {
    type F<A, B> = unsafe extern "C" fn(ObjcId, ObjcSel, A, B) -> ObjcId;
    let f: F<A, B> = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a, b)
}

/// Send a message with 3 args.
#[inline(always)]
pub unsafe fn msg3<A, B, C>(target: ObjcId, sel: ObjcSel, a: A, b: B, c: C) -> ObjcId {
    type F<A, B, C> = unsafe extern "C" fn(ObjcId, ObjcSel, A, B, C) -> ObjcId;
    let f: F<A, B, C> = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a, b, c)
}

// ── Void msg_send helpers (encoder hot path) ──

/// Send a message with 1 ObjcId arg, returning nothing.
#[inline(always)]
pub unsafe fn msg1_void(target: ObjcId, sel: ObjcSel, a: ObjcId) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a);
}

/// Send a message with 3 args (ObjcId, usize, usize), returning nothing.
#[inline(always)]
pub unsafe fn msg3_void(target: ObjcId, sel: ObjcSel, a: ObjcId, b: NSUInteger, c: NSUInteger) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger, NSUInteger);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, a, b, c);
}

/// Send a message with (ptr, usize, usize) args, returning nothing.
#[inline(always)]
pub unsafe fn msg_bytes_void(
    target: ObjcId,
    sel: ObjcSel,
    ptr: *const c_void,
    len: NSUInteger,
    index: NSUInteger,
) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, *const c_void, NSUInteger, NSUInteger);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, ptr, len, index);
}

/// Send a message with two MTLSize args, returning nothing.
#[inline(always)]
pub unsafe fn msg_dispatch_void(target: ObjcId, sel: ObjcSel, grid: MTLSize, group: MTLSize) {
    type F = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);
    let f: F = std::mem::transmute(objc_msgSend as *const c_void);
    f(target, sel, grid, group);
}

// ── Convenience wrappers (for rare/dynamic selectors — NOT hot path) ──

pub unsafe fn msg_send_0(target: ObjcId, selector: &str) -> ObjcId {
    msg0(target, sel(selector))
}

pub unsafe fn msg_send_1<A>(target: ObjcId, selector: &str, a: A) -> ObjcId {
    msg1(target, sel(selector), a)
}

pub unsafe fn msg_send_2<A, B>(target: ObjcId, selector: &str, a: A, b: B) -> ObjcId {
    msg2(target, sel(selector), a, b)
}

pub unsafe fn msg_send_3<A, B, C>(target: ObjcId, selector: &str, a: A, b: B, c: C) -> ObjcId {
    msg3(target, sel(selector), a, b, c)
}

pub unsafe fn msg_send_bool(target: ObjcId, selector: &str) -> bool {
    msg0_bool(target, sel(selector))
}

pub unsafe fn msg_send_usize(target: ObjcId, selector: &str) -> NSUInteger {
    msg0_usize(target, sel(selector))
}

pub unsafe fn msg_send_ptr(target: ObjcId, selector: &str) -> *mut c_void {
    msg0_ptr(target, sel(selector))
}

// ── String helpers ──

pub fn nserror_string(err: ObjcId) -> Option<String> {
    if err.is_null() {
        return None;
    }
    unsafe {
        let desc = msg0(err, SEL_localizedDescription());
        if desc.is_null() {
            return None;
        }
        nsstring_to_rust(desc)
    }
}

pub fn nsstring_to_rust(obj: ObjcId) -> Option<String> {
    if obj.is_null() {
        return None;
    }
    unsafe {
        type F = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
        let f: F = std::mem::transmute(objc_msgSend as *const c_void);
        let cstr = f(obj, SEL_UTF8String());
        if cstr.is_null() {
            return None;
        }
        Some(CStr::from_ptr(cstr).to_string_lossy().into_owned())
    }
}

/// Create NSString from &str. Zero-alloc: uses CFStringCreateWithBytes
/// directly on the Rust string's UTF-8 bytes, no CString intermediate.
pub fn nsstring(s: &str) -> ObjcId {
    unsafe {
        CFStringCreateWithBytes(
            std::ptr::null(),
            s.as_ptr(),
            s.len() as isize,
            kCFStringEncodingUTF8,
            false,
        ) as ObjcId
    }
}

// ── Retain/release (hot path — cached selectors) ──

/// Retain via objc_retain (direct C call, not msg_send).
#[inline(always)]
pub unsafe fn retain(obj: ObjcId) -> ObjcId {
    if !obj.is_null() {
        objc_retain(obj);
    }
    obj
}

/// Release via objc_release (direct C call, not msg_send).
#[inline(always)]
pub unsafe fn release(obj: ObjcId) {
    if !obj.is_null() {
        objc_release(obj);
    }
}

/// Retain a known-non-null object. No null check — hot path only.
#[inline(always)]
pub unsafe fn retain_nonnull(obj: ObjcId) -> ObjcId {
    debug_assert!(!obj.is_null());
    objc_retain(obj);
    obj
}

/// Release a known-non-null object. No null check — hot path only.
#[inline(always)]
pub unsafe fn release_nonnull(obj: ObjcId) {
    debug_assert!(!obj.is_null());
    objc_release(obj);
}
