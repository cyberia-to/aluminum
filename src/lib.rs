//! `metal` — Pure Rust access to Apple Metal GPU
//!
//! Direct C-level access to Metal.framework via objc_msgSend FFI.
//! Zero external dependencies. Only requires macOS with Metal support.
//!
//! ```no_run
//! use metal::{MtlDevice, MetalError};
//!
//! let device = MtlDevice::system_default()?;
//! let queue = device.new_command_queue()?;
//!
//! let source = r#"
//!     #include <metal_stdlib>
//!     kernel void add(device float *a [[buffer(0)]],
//!                     device float *b [[buffer(1)]],
//!                     device float *c [[buffer(2)]],
//!                     uint id [[thread_position_in_grid]]) {
//!         c[id] = a[id] + b[id];
//!     }
//! "#;
//!
//! let lib = device.new_library_with_source(source)?;
//! let func = lib.get_function("add")?;
//! let pipeline = device.new_compute_pipeline(&func)?;
//!
//! let n = 1024usize;
//! let buf_a = device.new_buffer(n * 4)?;
//! let buf_b = device.new_buffer(n * 4)?;
//! let buf_c = device.new_buffer(n * 4)?;
//!
//! let cmd = queue.command_buffer()?;
//! let enc = cmd.compute_encoder()?;
//! enc.set_pipeline(&pipeline);
//! enc.set_buffer(&buf_a, 0, 0);
//! enc.set_buffer(&buf_b, 0, 1);
//! enc.set_buffer(&buf_c, 0, 2);
//! enc.dispatch_threads((n, 1, 1), (64, 1, 1));
//! enc.end_encoding();
//! cmd.commit();
//! cmd.wait_until_completed();
//! # Ok::<(), metal::MetalError>(())
//! ```

#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    clippy::missing_transmute_annotations
)]

pub mod buffer;
pub mod command;
pub mod device;
pub mod dispatch;
pub mod encoder;
pub mod ffi;
pub mod pipeline;
pub mod shader;
pub mod sync;
pub mod texture;

pub use buffer::MtlBuffer;
pub use command::{MtlCommandBuffer, MtlCommandQueue};
pub use device::MtlDevice;
pub use dispatch::{BatchEncoder, ComputeDispatcher, GpuFuture};
pub use encoder::{MtlBlitEncoder, MtlComputeEncoder};
pub use pipeline::{MtlComputePipeline, MtlRenderPipeline};
pub use shader::{MtlFunction, MtlLibrary};
pub use sync::{MtlEvent, MtlFence, MtlSharedEvent};
pub use texture::MtlTexture;

pub use buffer::{cvt_f16_f32, cvt_f32_f16, f32_to_fp16, fp16_to_f32};

/// Execute a closure inside an autorelease pool.
/// Use this to scope autoreleased ObjC objects (e.g. from `command_buffer_unretained`).
#[inline]
pub fn autorelease_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    unsafe {
        let pool = ffi::objc_autoreleasePoolPush();
        let result = f();
        ffi::objc_autoreleasePoolPop(pool);
        result
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum MetalError {
    DeviceNotFound,
    BufferCreationFailed(String),
    LibraryCompilationFailed(String),
    FunctionNotFound(String),
    PipelineCreationFailed(String),
    CommandBufferError(String),
    EncoderCreationFailed,
    QueueCreationFailed,
    TextureCreationFailed(String),
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
