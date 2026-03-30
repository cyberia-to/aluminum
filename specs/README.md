# aluminium — API specification

pure Rust driver for Apple Metal GPU. direct objc_msgSend FFI,
zero external dependencies, only macOS system frameworks.

## concepts

| concept | what it is |
|---------|-----------|
| device | a Metal GPU — discovered at runtime, owns all GPU resources |
| buffer | CPU/GPU memory region — shared (zero-copy) or private (GPU-only) |
| library | compiled shader code — one or more functions from MSL source |
| function | a single shader entry point — vertex, fragment, or kernel |
| pipeline | a compiled GPU state object — binds function + config for dispatch |
| queue | a serial command submission channel to the GPU |
| command buffer | a batch of encoded GPU commands — submitted atomically |
| encoder | records commands into a command buffer — compute or blit |
| dispatcher | pre-resolved IMP dispatch engine for inference hot loops |
| texture | GPU image data — 2D/3D, region read/write |
| fence | GPU work tracking within a single command buffer |
| event | synchronization between command buffers |
| shared event | CPU/GPU synchronization with signaled counter |

## lifecycle

```
source  ->  compile  ->  pipeline  ->  encode  ->  commit  ->  complete
  MSL      MTLLibrary   pipeline    encoder    cmdBuf      GPU done
```

## device

| method | signature | semantics |
|--------|-----------|-----------|
| system_default | `() -> Result<MtlDevice>` | get default Metal GPU |
| all | `() -> Result<Vec<MtlDevice>>` | enumerate all Metal GPUs |
| name | `(&self) -> String` | device name (e.g. "Apple M1 Pro") |
| has_unified_memory | `(&self) -> bool` | shared CPU/GPU memory architecture |
| max_buffer_length | `(&self) -> usize` | max buffer allocation in bytes |
| max_threads_per_threadgroup | `(&self) -> MTLSize` | max threads per threadgroup |
| recommended_max_working_set_size | `(&self) -> u64` | recommended GPU memory budget |
| new_command_queue | `(&self) -> Result<MtlCommandQueue>` | create command queue |
| new_buffer | `(&self, bytes) -> Result<MtlBuffer>` | allocate shared buffer (CPU+GPU) |
| new_buffer_private | `(&self, bytes) -> Result<MtlBuffer>` | allocate private buffer (GPU-only) |
| new_buffer_with_data | `(&self, &[u8]) -> Result<MtlBuffer>` | shared buffer with initial data |
| new_library_with_source | `(&self, &str) -> Result<MtlLibrary>` | compile MSL source |
| new_compute_pipeline | `(&self, &MtlFunction) -> Result<MtlComputePipeline>` | create compute pipeline |
| new_texture | `(&self, desc) -> Result<MtlTexture>` | create texture from descriptor (unsafe) |
| new_fence | `(&self) -> MtlFence` | create fence |
| new_event | `(&self) -> MtlEvent` | create event |
| new_shared_event | `(&self) -> MtlSharedEvent` | create shared event |

### apple mapping

| method | ObjC |
|--------|------|
| system_default | MTLCreateSystemDefaultDevice() |
| all | MTLCopyAllDevices() |
| name | [device name] |
| has_unified_memory | [device hasUnifiedMemory] |
| max_buffer_length | [device maxBufferLength] |
| max_threads_per_threadgroup | [device maxThreadsPerThreadgroup] |
| recommended_max_working_set_size | [device recommendedMaxWorkingSetSize] |
| new_command_queue | [device newCommandQueue] |
| new_buffer | [device newBufferWithLength:options:] (StorageModeShared) |
| new_buffer_private | [device newBufferWithLength:options:] (StorageModePrivate) |
| new_buffer_with_data | [device newBufferWithBytes:length:options:] |
| new_library_with_source | [device newLibraryWithSource:options:error:] |
| new_compute_pipeline | [device newComputePipelineStateWithFunction:error:] |
| new_texture | [device newTextureWithDescriptor:] |
| new_fence | [device newFence] |
| new_event | [device newEvent] |
| new_shared_event | [device newSharedEvent] |

## buffer

CPU/GPU memory region. two storage modes:

- **shared** (default) — zero-copy, CPU and GPU share physical memory.
  no lock/unlock needed. contents pointer cached at creation.
- **private** — GPU-only, higher bandwidth for inter-kernel buffers.
  CPU cannot read/write. use blit encoder to copy data in/out.

| method | signature | semantics |
|--------|-----------|-----------|
| is_shared | `(&self) -> bool` | true if CPU-accessible (shared mode) |
| contents | `(&self) -> *mut c_void` | raw pointer to shared memory (cached) |
| with_data | `(&self, \|&[u8]\|)` | read access via closure |
| with_data_mut | `(&self, \|&mut [u8]\|)` | write access via closure |
| with_f32 | `(&self, \|&[f32]\|)` | typed read as f32 |
| with_f32_mut | `(&self, \|&mut [f32]\|)` | typed write as f32 |
| size | `(&self) -> usize` | allocation in bytes |
| drop | automatic | [buffer release] |

### apple mapping

| method | ObjC |
|--------|------|
| contents | [buffer contents] |
| size | construction parameter |
| drop | objc_release |

## library

compiled shader code from MSL source text.

| method | signature | semantics |
|--------|-----------|-----------|
| get_function | `(&self, &str) -> Result<MtlFunction>` | get function by name |
| function_names | `(&self) -> Vec<String>` | list all function names |

### apple mapping

| method | ObjC |
|--------|------|
| get_function | [library newFunctionWithName:] |
| function_names | [library functionNames] |

## function

a single shader entry point extracted from a library.

| method | signature | semantics |
|--------|-----------|-----------|
| name | `(&self) -> String` | function name |

## compute pipeline

compiled GPU state — function + hardware config.

| method | signature | semantics |
|--------|-----------|-----------|
| max_total_threads_per_threadgroup | `(&self) -> usize` | max threads per threadgroup for this pipeline |
| thread_execution_width | `(&self) -> usize` | SIMD width (32 on Apple GPU) |
| static_threadgroup_memory_length | `(&self) -> usize` | threadgroup memory used by pipeline (bytes) |

### apple mapping

| method | ObjC |
|--------|------|
| max_total_threads_per_threadgroup | [pipeline maxTotalThreadsPerThreadgroup] |
| thread_execution_width | [pipeline threadExecutionWidth] |
| static_threadgroup_memory_length | [pipeline staticThreadgroupMemoryLength] |

## command queue

| method | signature | semantics |
|--------|-----------|-----------|
| command_buffer | `(&self) -> Result<MtlCommandBuffer>` | retained, ARC fast-retain |
| command_buffer_unretained | `unsafe (&self) -> Result<MtlCommandBuffer>` | autoreleased, no retain overhead |
| command_buffer_fast | `unsafe (&self) -> Result<MtlCommandBuffer>` | unretained references — Metal skips resource retain/release |
| command_buffer_unchecked | `unsafe (&self) -> MtlCommandBuffer` | unretained refs, no null check |
| command_buffer_autoreleased | `unsafe (&self) -> MtlCommandBuffer` | fastest — must be in autorelease_pool |

overhead hierarchy (low to high):

```
command_buffer_autoreleased   — zero overhead, requires pool
command_buffer_unchecked      — no null check, unretained refs
command_buffer_fast           — unretained refs, null checked
command_buffer_unretained     — autoreleased, null checked
command_buffer                — retained, safe, standard
```

### apple mapping

| method | ObjC |
|--------|------|
| command_buffer | [queue commandBuffer] + objc_retainAutoreleasedReturnValue |
| command_buffer_unretained | [queue commandBuffer] (no retain) |
| command_buffer_fast | [queue commandBufferWithUnretainedReferences] + retain |
| command_buffer_unchecked | [queue commandBufferWithUnretainedReferences] + ARC fast-retain |
| command_buffer_autoreleased | [queue commandBufferWithUnretainedReferences] (no retain) |

## command buffer

| method | signature | semantics |
|--------|-----------|-----------|
| compute_encoder | `(&self) -> Result<MtlComputeEncoder>` | retained compute encoder |
| compute_encoder_unretained | `unsafe (&self) -> Result<MtlComputeEncoder>` | autoreleased |
| compute_encoder_unchecked | `unsafe (&self) -> MtlComputeEncoder` | no null check, retained |
| compute_encoder_autoreleased | `unsafe (&self) -> MtlComputeEncoder` | fastest, requires pool |
| blit_encoder | `(&self) -> Result<MtlBlitEncoder>` | blit encoder |
| commit | `(&self)` | submit for GPU execution |
| wait_until_completed | `(&self)` | block until GPU done |
| status | `(&self) -> u64` | execution status code |
| error | `(&self) -> Option<String>` | error description if failed |
| gpu_start_time | `(&self) -> f64` | GPU start time (seconds since boot) |
| gpu_end_time | `(&self) -> f64` | GPU end time (seconds since boot) |
| gpu_time | `(&self) -> f64` | GPU execution duration (end - start) |

### apple mapping

| method | ObjC |
|--------|------|
| compute_encoder | [cmdBuf computeCommandEncoder] + ARC fast-retain |
| blit_encoder | [cmdBuf blitCommandEncoder] |
| commit | [cmdBuf commit] |
| wait_until_completed | [cmdBuf waitUntilCompleted] |
| status | [cmdBuf status] |
| error | [cmdBuf error] |
| gpu_start_time | [cmdBuf GPUStartTime] |
| gpu_end_time | [cmdBuf GPUEndTime] |

## compute encoder

| method | signature | semantics |
|--------|-----------|-----------|
| set_pipeline | `(&self, &MtlComputePipeline)` | bind compute pipeline |
| set_buffer | `(&self, &MtlBuffer, offset, index)` | bind buffer at index |
| set_bytes | `(&self, &[u8], index)` | inline constant data |
| dispatch_threads | `(&self, grid, group)` | dispatch with auto non-uniform grid handling |
| dispatch_threadgroups | `(&self, groups, threads)` | dispatch with explicit group count |
| end_encoding | `(&self)` | finish encoding |

### apple mapping

| method | ObjC |
|--------|------|
| set_pipeline | [encoder setComputePipelineState:] |
| set_buffer | [encoder setBuffer:offset:atIndex:] |
| set_bytes | [encoder setBytes:length:atIndex:] |
| dispatch_threads | [encoder dispatchThreads:threadsPerThreadgroup:] |
| dispatch_threadgroups | [encoder dispatchThreadgroups:threadsPerThreadgroup:] |
| end_encoding | [encoder endEncoding] |

## blit encoder

| method | signature | semantics |
|--------|-----------|-----------|
| copy_buffer | `(&self, src, src_off, dst, dst_off, size)` | GPU buffer-to-buffer copy |
| end_encoding | `(&self)` | finish encoding |

### apple mapping

| method | ObjC |
|--------|------|
| copy_buffer | [encoder copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:] |
| end_encoding | [encoder endEncoding] |

## compute dispatcher

pre-resolved IMP dispatch engine for inference hot loops.
resolves all ObjC method implementations at construction — every
dispatch call goes through direct function pointers, bypassing
objc_msgSend entirely.

| method | signature | semantics |
|--------|-----------|-----------|
| new | `(&MtlCommandQueue) -> Self` | resolve all IMPs eagerly |
| dispatch | `unsafe (&self, pipeline, buffers, grid, group)` | single dispatch: encode + commit + wait |
| dispatch_with_bytes | `unsafe (&self, pipeline, buffers, bytes, index, grid, group)` | single dispatch with inline constants |
| dispatch_batch | `unsafe (&self, \|&BatchEncoder\|)` | multiple dispatches in one command buffer |
| dispatch_batch_raw | `unsafe (&self, \|&BatchEncoder\|)` | batch without autorelease management (caller manages pool) |
| dispatch_batch_async | `unsafe (&self, \|&BatchEncoder\|) -> GpuFuture` | encode + commit, return handle for deferred wait |

### batch encoder

provided to dispatch_batch closures. same IMP-resolved hot path.

| method | signature | semantics |
|--------|-----------|-----------|
| set_pipeline | `(&self, &MtlComputePipeline)` | bind pipeline |
| set_buffer | `(&self, &MtlBuffer, offset, index)` | bind buffer |
| set_bytes | `(&self, &[u8], index)` | inline constants |
| dispatch_threads | `(&self, grid, group)` | dispatch |
| dispatch_threadgroups | `(&self, groups, threads)` | dispatch with explicit groups |

### gpu future

handle for committed but not yet completed command buffer.

| method | signature | semantics |
|--------|-----------|-----------|
| wait | `(self)` | block until GPU finishes, release command buffer |
| drop | automatic | if not waited, waits + releases (prevents leak) |

### pipelining pattern

```rust
let mut prev = None;
for pass in passes {
    let handle = disp.dispatch_batch_async(|batch| { ... });
    if let Some(h) = prev { h.wait(); }
    prev = Some(handle);
}
if let Some(h) = prev { h.wait(); }
```

overlap GPU execution of batch N with CPU encoding of batch N+1.

## texture

GPU image data. wraps `id<MTLTexture>`.

| method | signature | semantics |
|--------|-----------|-----------|
| width | `(&self) -> usize` | width in pixels |
| height | `(&self) -> usize` | height in pixels |
| depth | `(&self) -> usize` | depth (3D textures) |
| pixel_format | `(&self) -> usize` | MTLPixelFormat value |
| replace_region | `unsafe (&self, region, mipmap, data, bytes_per_row)` | write data to region |
| get_bytes | `unsafe (&self, data, bytes_per_row, region, mipmap)` | read data from region |

### apple mapping

| method | ObjC |
|--------|------|
| width | [texture width] |
| height | [texture height] |
| depth | [texture depth] |
| pixel_format | [texture pixelFormat] |
| replace_region | [texture replaceRegion:mipmapLevel:withBytes:bytesPerRow:] |
| get_bytes | [texture getBytes:bytesPerRow:fromRegion:mipmapLevel:] |

## synchronization

### fence

GPU work tracking within a single command buffer.

| method | signature | semantics |
|--------|-----------|-----------|
| as_raw | `(&self) -> ObjcId` | raw pointer for encoder fence ops |

### event

synchronization between command buffers on same device.

| method | signature | semantics |
|--------|-----------|-----------|
| as_raw | `(&self) -> ObjcId` | raw pointer for command buffer signal/wait |

### shared event

CPU/GPU synchronization with monotonic counter.

| method | signature | semantics |
|--------|-----------|-----------|
| signaled_value | `(&self) -> u64` | current signaled counter value |
| as_raw | `(&self) -> ObjcId` | raw pointer |

## conversion

fp16<->f32 via inline NEON assembly (aarch64) with software fallback.

| function | signature | semantics |
|----------|-----------|-----------|
| fp16_to_f32 | `(u16) -> f32` | single half -> single precision |
| f32_to_fp16 | `(f32) -> u16` | single -> half precision |
| cvt_f16_f32 | `(&mut [f32], &[u16])` | bulk half -> single (32/iter, 4x unrolled NEON) |
| cvt_f32_f16 | `(&mut [u16], &[f32])` | bulk single -> half (32/iter, 4x unrolled NEON) |

tail: 8/iter NEON, then scalar fallback.

## autorelease pool

```rust
autorelease_pool(|| {
    // autoreleased ObjC objects valid here
})
```

required when using unretained/autoreleased command buffer and encoder variants.

## errors

```
DeviceNotFound              no Metal GPU available
BufferCreationFailed(String) buffer allocation failed
LibraryCompilationFailed(String) MSL compilation error
FunctionNotFound(String)    shader function not in library
PipelineCreationFailed(String) pipeline creation error
CommandBufferError(String)  command buffer execution error
EncoderCreationFailed       encoder creation failed
QueueCreationFailed         command queue creation failed
TextureCreationFailed(String) texture creation failed
Io(io::Error)               filesystem error
```

## execution model

- one pipeline = one compiled shader function
- command buffers submitted atomically via commit
- GPU executes command buffers in order per queue
- multiple queues enable concurrent GPU work
- shared buffers need no synchronization between command buffer boundaries
- private buffers need blit encoder for CPU data transfer
- dispatch_threads handles non-uniform grids automatically
- dispatch_threadgroups requires manual grid division
- ComputeDispatcher bypasses objc_msgSend for hot-loop performance

## driver stack

```
aluminium crate (objc_msgSend FFI + IMP resolution)
  -> Metal.framework (linked at build time)
    -> GPU driver
      -> GPU hardware
```

Metal.framework is public. linked via `#[link(name = "Metal", kind = "framework")]`.
core path: objc_msgSend with transmuted function pointers.
hot path: pre-resolved IMP via class_getMethodImplementation.

## render pipeline

`MtlRenderPipeline` struct exists (wraps `id<MTLRenderPipelineState>`).
no render encoder yet — compute-only driver.
