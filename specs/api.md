# metal — API specification

the public interface for Apple Metal GPU access from Rust.

## concepts

| concept | what it is |
|---------|-----------|
| device | a Metal GPU — discovered at runtime, owns all GPU resources |
| buffer | shared-memory byte region — the zero-copy CPU↔GPU data path |
| library | compiled shader code — one or more functions from MSL source |
| function | a single shader entry point — vertex, fragment, or kernel |
| pipeline | a compiled GPU state object — binds function + config for dispatch |
| queue | a serial command submission channel to the GPU |
| command buffer | a batch of encoded GPU commands — submitted atomically |
| encoder | records commands into a command buffer — compute, blit, or render |

## lifecycle

```
source  →  compile  →  pipeline  →  encode  →  commit  →  complete
  MSL      MTLLibrary   pipeline    encoder    cmdBuf      GPU done
```

compile: MSL source → MTLLibrary (GPU bytecode)
pipeline: MTLFunction → MTLComputePipelineState
encode: bind pipeline + resources, dispatch threads
commit: submit command buffer to GPU
complete: GPU finishes, results in shared buffers

## device

| method | signature | semantics |
|--------|-----------|-----------|
| system_default | `() → Result<Device>` | get default Metal GPU |
| all | `() → Result<Vec<Device>>` | enumerate all Metal GPUs |
| name | `(&self) → String` | device name |
| has_unified_memory | `(&self) → bool` | shared CPU/GPU memory |
| max_buffer_length | `(&self) → usize` | max buffer allocation |
| new_command_queue | `(&self) → Result<Queue>` | create command queue |
| new_buffer | `(&self, bytes) → Result<Buffer>` | allocate shared buffer |
| new_buffer_with_data | `(&self, &[u8]) → Result<Buffer>` | buffer from data |
| new_library_with_source | `(&self, &str) → Result<Library>` | compile MSL |
| new_compute_pipeline | `(&self, &Function) → Result<ComputePipeline>` | create pipeline |

### apple mapping

| method | ObjC selector |
|--------|--------------|
| system_default | MTLCreateSystemDefaultDevice() |
| all | MTLCopyAllDevices() |
| name | [device name] |
| new_command_queue | [device newCommandQueue] |
| new_buffer | [device newBufferWithLength:options:] |
| new_buffer_with_data | [device newBufferWithBytes:length:options:] |
| new_library_with_source | [device newLibraryWithSource:options:error:] |
| new_compute_pipeline | [device newComputePipelineStateWithFunction:error:] |

## buffer

shared-memory byte region. zero-copy between CPU and GPU.
uses MTLResourceStorageModeShared — no lock/unlock needed.

| method | signature | semantics |
|--------|-----------|-----------|
| contents | `(&self) → *mut c_void` | raw pointer to shared memory |
| with_data | `(&self, \|&[u8]\|)` | read access via closure |
| with_data_mut | `(&self, \|&mut [u8]\|)` | write access via closure |
| with_f32 | `(&self, \|&[f32]\|)` | typed read as f32 |
| with_f32_mut | `(&self, \|&mut [f32]\|)` | typed write as f32 |
| size | `(&self) → usize` | allocation in bytes |
| drop | automatic | [buffer release] |

### apple mapping

| method | ObjC selector |
|--------|--------------|
| new_buffer | [device newBufferWithLength:options:] |
| contents | [buffer contents] |
| drop | [buffer release] |

## library

compiled shader code from MSL source text.

| method | signature | semantics |
|--------|-----------|-----------|
| get_function | `(&self, &str) → Result<Function>` | get function by name |
| function_names | `(&self) → Vec<String>` | list all function names |

### apple mapping

| method | ObjC selector |
|--------|--------------|
| get_function | [library newFunctionWithName:] |
| function_names | [library functionNames] |

## command queue + buffer

| method | signature | semantics |
|--------|-----------|-----------|
| command_buffer | `(&self) → Result<CmdBuf>` | create command buffer |
| compute_encoder | `(&self) → Result<Encoder>` | create compute encoder |
| blit_encoder | `(&self) → Result<BlitEncoder>` | create blit encoder |
| commit | `(&self)` | submit for execution |
| wait_until_completed | `(&self)` | block until GPU done |
| status | `(&self) → u64` | execution status |
| error | `(&self) → Option<String>` | error description |

### apple mapping

| method | ObjC selector |
|--------|--------------|
| command_buffer | [queue commandBuffer] |
| compute_encoder | [cmdBuf computeCommandEncoder] |
| commit | [cmdBuf commit] |
| wait_until_completed | [cmdBuf waitUntilCompleted] |

## compute encoder

| method | signature | semantics |
|--------|-----------|-----------|
| set_pipeline | `(&self, &Pipeline)` | bind compute pipeline |
| set_buffer | `(&self, &Buffer, offset, index)` | bind buffer at index |
| set_bytes | `(&self, &[u8], index)` | inline constant data |
| dispatch_threads | `(&self, grid, group)` | dispatch with auto-sizing |
| dispatch_threadgroups | `(&self, groups, threads)` | dispatch with explicit groups |
| end_encoding | `(&self)` | finish encoding |

### apple mapping

| method | ObjC selector |
|--------|--------------|
| set_pipeline | [encoder setComputePipelineState:] |
| set_buffer | [encoder setBuffer:offset:atIndex:] |
| set_bytes | [encoder setBytes:length:atIndex:] |
| dispatch_threads | [encoder dispatchThreads:threadsPerThreadgroup:] |
| end_encoding | [encoder endEncoding] |

## conversion

fp16↔f32 conversion via inline NEON assembly (ARM64) with software fallback.

| function | signature | semantics |
|----------|-----------|-----------|
| fp16_to_f32 | `(u16) → f32` | half → single precision |
| f32_to_fp16 | `(f32) → u16` | single → half precision |
| cvt_f16_f32 | `(&mut [f32], &[u16])` | bulk half → single (8 at a time) |
| cvt_f32_f16 | `(&mut [u16], &[f32])` | bulk single → half (8 at a time) |

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
- command buffers are submitted atomically via commit
- GPU executes command buffers in order per queue
- multiple queues enable concurrent GPU work
- buffers with StorageModeShared need no synchronization for CPU/GPU access between command buffer boundaries
- dispatch_threads handles non-uniform grids automatically
- dispatch_threadgroups requires manual grid division

## driver stack

```
metal crate (objc_msgSend FFI)
  → Metal.framework (linked at build time)
    → GPU driver
      → GPU hardware
```

Metal.framework is public. linked via `#[link(name = "Metal", kind = "framework")]`.
all protocol methods via objc_msgSend with transmuted function pointers.
