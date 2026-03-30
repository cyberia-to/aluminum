#![doc = include_str!("../README.md")]
#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    clippy::missing_transmute_annotations
)]

#[cfg(test)]
mod tests;

pub mod buffer;
pub mod command;
pub mod device;
pub mod dispatch;
pub mod encoder;
pub mod ffi;
pub mod fp16;
pub mod pipeline;
pub mod shader;
pub mod sync;
pub mod texture;

pub use buffer::MtlBuffer;
pub use command::{MtlCommandBuffer, MtlCommandQueue};
pub use device::MtlDevice;
pub use dispatch::{BatchEncoder, ComputeDispatcher, GpuFuture};
pub use encoder::{MtlBlitEncoder, MtlComputeEncoder};
pub use pipeline::MtlComputePipeline;
pub use shader::{MtlFunction, MtlLibrary};
pub use sync::{MtlEvent, MtlFence, MtlSharedEvent};
pub use texture::MtlTexture;

pub use fp16::{cvt_f16_f32, cvt_f32_f16, f32_to_fp16, fp16_to_f32};

/// Execute a closure inside an autorelease pool.
/// Use this to scope autoreleased ObjC objects (e.g. from `command_buffer_unretained`).
/// The pool is drained even if the closure panics (unwind-safe).
#[inline]
pub fn autorelease_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct PoolGuard(*mut std::ffi::c_void);
    impl Drop for PoolGuard {
        fn drop(&mut self) {
            unsafe { ffi::objc_autoreleasePoolPop(self.0) };
        }
    }
    let pool = unsafe { ffi::objc_autoreleasePoolPush() };
    let _guard = PoolGuard(pool);
    f()
}

/// Errors returned by aruminium operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum MetalError {
    /// No Metal-capable GPU found on this system.
    DeviceNotFound,
    /// GPU buffer allocation failed.
    BufferCreationFailed(String),
    /// MSL shader compilation failed (includes compiler diagnostic).
    LibraryCompilationFailed(String),
    /// Named function not found in compiled shader library.
    FunctionNotFound(String),
    /// Compute pipeline creation failed.
    PipelineCreationFailed(String),
    /// Command buffer execution failed.
    CommandBufferError(String),
    /// Could not create a command encoder.
    EncoderCreationFailed,
    /// Could not create a command queue.
    QueueCreationFailed,
    /// Texture creation failed.
    TextureCreationFailed(String),
    /// Filesystem I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::DeviceNotFound => write!(f, "No Metal device found"),
            MetalError::BufferCreationFailed(msg) => write!(f, "Buffer creation failed: {}", msg),
            MetalError::LibraryCompilationFailed(msg) => {
                write!(f, "Shader compilation failed: {}", msg)
            }
            MetalError::FunctionNotFound(name) => write!(f, "Function not found: {}", name),
            MetalError::PipelineCreationFailed(msg) => {
                write!(f, "Pipeline creation failed: {}", msg)
            }
            MetalError::CommandBufferError(msg) => write!(f, "Command buffer error: {}", msg),
            MetalError::EncoderCreationFailed => write!(f, "Encoder creation failed"),
            MetalError::QueueCreationFailed => write!(f, "Command queue creation failed"),
            MetalError::TextureCreationFailed(msg) => {
                write!(f, "Texture creation failed: {}", msg)
            }
            MetalError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for MetalError {}

impl From<std::io::Error> for MetalError {
    fn from(e: std::io::Error) -> Self {
        MetalError::Io(e)
    }
}
