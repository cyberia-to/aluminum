//! MtlDevice: Metal GPU device discovery and factory methods

use crate::buffer::MtlBuffer;
use crate::command::MtlCommandQueue;
use crate::ffi::*;
use crate::pipeline::MtlComputePipeline;
use crate::shader::{MtlFunction, MtlLibrary};
use crate::texture::MtlTexture;
use crate::MetalError;
use std::ffi::c_void;

/// A Metal GPU device. Wraps `id<MTLDevice>`.
pub struct MtlDevice {
    raw: ObjcId,
}

impl MtlDevice {
    /// Get the system default Metal device.
    pub fn system_default() -> Result<Self, MetalError> {
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        if raw.is_null() {
            return Err(MetalError::DeviceNotFound);
        }
        Ok(MtlDevice { raw })
    }

    /// Get all available Metal devices.
    pub fn all() -> Result<Vec<Self>, MetalError> {
        let arr = unsafe { MTLCopyAllDevices() };
        if arr.is_null() {
            return Err(MetalError::DeviceNotFound);
        }
        let count = unsafe { msg0_usize(arr, SEL_count()) };
        let mut devices = Vec::with_capacity(count);
        for i in 0..count {
            let dev = unsafe { msg1::<NSUInteger>(arr, SEL_objectAtIndex(), i) };
            if !dev.is_null() {
                unsafe { retain(dev) };
                devices.push(MtlDevice { raw: dev });
            }
        }
        unsafe { release(arr) };
        Ok(devices)
    }

    /// Device name (e.g. "Apple M1 Pro").
    pub fn name(&self) -> String {
        unsafe {
            let ns = msg0(self.raw, SEL_name());
            nsstring_to_rust(ns).unwrap_or_else(|| "unknown".into())
        }
    }

    /// Whether the device has unified memory (shared CPU/GPU).
    pub fn has_unified_memory(&self) -> bool {
        unsafe { msg0_bool(self.raw, SEL_hasUnifiedMemory()) }
    }

    /// Maximum buffer length in bytes.
    pub fn max_buffer_length(&self) -> usize {
        unsafe { msg0_usize(self.raw, SEL_maxBufferLength()) }
    }

    /// Maximum threads per threadgroup.
    pub fn max_threads_per_threadgroup(&self) -> MTLSize {
        unsafe { msg0_mtlsize(self.raw, SEL_maxThreadsPerThreadgroup()) }
    }

    /// Recommended max working set size in bytes.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        unsafe { msg0_u64(self.raw, SEL_recommendedMaxWorkingSetSize()) }
    }

    /// Create a new command queue.
    pub fn new_command_queue(&self) -> Result<MtlCommandQueue, MetalError> {
        let raw = unsafe { msg0(self.raw, SEL_newCommandQueue()) };
        if raw.is_null() {
            return Err(MetalError::QueueCreationFailed);
        }
        Ok(MtlCommandQueue::from_raw(raw))
    }

    /// Create a shared-mode buffer of the given byte size.
    ///
    /// `size` must be > 0. Use `new_buffer_with_data` for initialized buffers.
    pub fn new_buffer(&self, size: usize) -> Result<MtlBuffer, MetalError> {
        if size == 0 {
            return Err(MetalError::BufferCreationFailed("size must be > 0".into()));
        }
        let raw = unsafe {
            msg2::<NSUInteger, NSUInteger>(
                self.raw,
                SEL_newBufferWithLength_options(),
                size,
                MTLResourceStorageModeShared,
            )
        };
        if raw.is_null() {
            return Err(MetalError::BufferCreationFailed(format!("{} bytes", size)));
        }
        Ok(MtlBuffer::from_raw(raw, size))
    }

    /// Create a GPU-private buffer. Not accessible from CPU — use blit encoder to copy.
    /// Private storage gives Metal full control over memory placement and caching,
    /// which can yield higher GPU bandwidth for inter-layer inference buffers.
    pub fn new_buffer_private(&self, size: usize) -> Result<MtlBuffer, MetalError> {
        if size == 0 {
            return Err(MetalError::BufferCreationFailed("size must be > 0".into()));
        }
        let raw = unsafe {
            msg2::<NSUInteger, NSUInteger>(
                self.raw,
                SEL_newBufferWithLength_options(),
                size,
                MTLResourceStorageModePrivate,
            )
        };
        if raw.is_null() {
            return Err(MetalError::BufferCreationFailed(format!("{} bytes", size)));
        }
        Ok(MtlBuffer::from_raw_private(raw, size))
    }

    /// Create a shared-mode buffer initialized with data.
    pub fn new_buffer_with_data(&self, data: &[u8]) -> Result<MtlBuffer, MetalError> {
        let raw = unsafe {
            msg3::<*const c_void, NSUInteger, NSUInteger>(
                self.raw,
                SEL_newBufferWithBytes_length_options(),
                data.as_ptr() as *const c_void,
                data.len(),
                MTLResourceStorageModeShared,
            )
        };
        if raw.is_null() {
            return Err(MetalError::BufferCreationFailed(format!(
                "{} bytes",
                data.len()
            )));
        }
        Ok(MtlBuffer::from_raw(raw, data.len()))
    }

    /// Compile Metal Shading Language source into a library.
    pub fn new_library_with_source(&self, source: &str) -> Result<MtlLibrary, MetalError> {
        let ns_source = nsstring(source);
        let mut error: ObjcId = std::ptr::null_mut();
        let raw = unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, ObjcId, *mut ObjcId) -> ObjcId;
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            let r = f(
                self.raw,
                SEL_newLibraryWithSource_options_error(),
                ns_source,
                std::ptr::null_mut(),
                &mut error,
            );
            CFRelease(ns_source as CFTypeRef);
            r
        };
        if raw.is_null() {
            let msg = nserror_string(error).unwrap_or_else(|| "unknown error".into());
            return Err(MetalError::LibraryCompilationFailed(msg));
        }
        Ok(MtlLibrary::from_raw(raw))
    }

    /// Create a compute pipeline from a function.
    pub fn new_compute_pipeline(
        &self,
        function: &MtlFunction,
    ) -> Result<MtlComputePipeline, MetalError> {
        let mut error: ObjcId = std::ptr::null_mut();
        let raw = unsafe {
            type F = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, *mut ObjcId) -> ObjcId;
            let f: F = std::mem::transmute(objc_msgSend as *const c_void);
            f(
                self.raw,
                SEL_newComputePipelineStateWithFunction_error(),
                function.as_raw(),
                &mut error,
            )
        };
        if raw.is_null() {
            let msg = nserror_string(error).unwrap_or_else(|| "unknown error".into());
            return Err(MetalError::PipelineCreationFailed(msg));
        }
        Ok(MtlComputePipeline::from_raw(raw))
    }

    /// Create a texture with the given descriptor.
    ///
    /// # Safety
    /// `desc` must be a valid MTLTextureDescriptor object.
    pub unsafe fn new_texture(&self, desc: ObjcId) -> Result<MtlTexture, MetalError> {
        let raw = msg1::<ObjcId>(self.raw, SEL_newTextureWithDescriptor(), desc);
        if raw.is_null() {
            return Err(MetalError::TextureCreationFailed(
                "descriptor rejected".into(),
            ));
        }
        Ok(MtlTexture::from_raw(raw))
    }

    /// Create a new fence.
    pub fn new_fence(&self) -> Result<crate::sync::MtlFence, MetalError> {
        let raw = unsafe { msg0(self.raw, SEL_newFence()) };
        if raw.is_null() {
            return Err(MetalError::CommandBufferError(
                "fence creation failed".into(),
            ));
        }
        Ok(crate::sync::MtlFence::from_raw(raw))
    }

    /// Create a new event.
    pub fn new_event(&self) -> Result<crate::sync::MtlEvent, MetalError> {
        let raw = unsafe { msg0(self.raw, SEL_newEvent()) };
        if raw.is_null() {
            return Err(MetalError::CommandBufferError(
                "event creation failed".into(),
            ));
        }
        Ok(crate::sync::MtlEvent::from_raw(raw))
    }

    /// Create a new shared event.
    pub fn new_shared_event(&self) -> Result<crate::sync::MtlSharedEvent, MetalError> {
        let raw = unsafe { msg0(self.raw, SEL_newSharedEvent()) };
        if raw.is_null() {
            return Err(MetalError::CommandBufferError(
                "shared event creation failed".into(),
            ));
        }
        Ok(crate::sync::MtlSharedEvent::from_raw(raw))
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlDevice {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
