# Claude Code Instructions

## auditor mindset

the project is supervised by an engineer with 30 years of experience.
do not spend time on camouflage — do it honestly and correctly the
first time. one time correctly is cheaper than five times beautifully.

## honesty

never fake results. if a system produces nothing — show nothing.
a dash is more honest than a copied number. never substitute
appearance of progress for actual progress. never generate placeholder
data to fill a gap.

## literal interpretation

when the user says something, they mean it literally. do not
reinterpret. if unsure, ask once. do not guess and iterate.

## chain of verification

for non-trivial decisions affecting correctness:
1. initial answer
2. 3-5 verification questions that would expose errors
3. answer each independently
4. revised answer incorporating corrections

skip for trivial tasks.

## build & verify

```bash
cargo fmt --all                           # format
cargo clippy --workspace -- -W warnings   # lint
cargo build --release --workspace         # build all
cargo run --example vecadd                # verify Metal access
```

every commit: format clean, clippy clean, builds, examples run.

## project: aluminium

pure Rust driver for Apple Metal GPU. compile shaders, create pipelines,
dispatch compute and render work on GPU hardware. zero external
dependencies in the core crate — only macOS system frameworks.

## architecture

cargo workspace with two members:

- `aluminium` (root crate) — library + metal_probe binary + examples.
  zero external dependencies. links only system frameworks via FFI.
- `tools/` (crate `metal-tools`) — CLI binaries (bench).
  heavy dependencies isolated here.

```
src/                  core library (zero deps)
  lib.rs              public API: MtlDevice, MtlBuffer, MetalError
  ffi.rs              Metal.framework, CoreFoundation, libobjc FFI
  device.rs           MtlDevice: discovery, properties, factory methods
  buffer.rs           MtlBuffer: zero-copy shared GPU memory, fp16 conversion
  command.rs          MtlCommandQueue, MtlCommandBuffer
  encoder.rs          MtlComputeEncoder, MtlBlitEncoder
  shader.rs           MtlLibrary, MtlFunction: MSL compilation
  pipeline.rs         MtlComputePipeline, MtlRenderPipeline
  sync.rs             MtlFence, MtlEvent, MtlSharedEvent
  texture.rs          MtlTexture
  probe/              metal_probe — 5-level capability probe
examples/             runnable demos (vecadd, matmul)
tools/                separate crate with heavy deps (bench)
specs/                specification (source of truth)
  api.md              API spec — concepts, lifecycle, apple mapping
```

## source of truth

`specs/` is canonical. if specs/ and code disagree, resolve
in specs/ first, then propagate to code.

## pipeline contract

```
MSL source → Metal.framework (linked)
  → MTLLibrary (compile → GPU bytecode)
    → MTLComputePipelineState
      → Command Buffer → Encoder → Dispatch
        → GPU hardware
```

MTLBuffer with StorageModeShared is the zero-copy CPU↔GPU data path.
no lock/unlock needed for shared mode.

## key gotchas

- Metal access requires macOS with Metal-capable GPU. examples fail on Linux.
- Metal.framework is public — linked at compile time, not dlopen.
- all ObjC protocol methods go through objc_msgSend transmuted to typed fn pointers.
- MTLSize struct (24 bytes) passed in registers on ARM64 — verify transmute works.
- newXxx methods return retained objects. commandBuffer is autoreleased — retain it.
- MTLBuffer.contents() returns raw pointer — valid for buffer lifetime.
- never write naive loops for matmul on CPU — use Accelerate.framework.

## do not touch

without explicit discussion:
- Cargo.toml dependency versions
- src/ffi.rs FFI signatures (must match system frameworks)
- specs/ structure
- LICENSE

## quality

file size limit: 500 lines per source file. split into submodules
if exceeded.

every commit:
- type check / lint — zero warnings
- builds clean
- examples run

## coding conventions

- no ObjC, no Swift, no headers. FFI through libobjc + Metal.framework link.
- MTLBuffer for zero-copy CPU↔GPU data. inline NEON asm for fp16.
- `cargo fmt` enforced. clippy clean.

## git workflow

- atomic commits — one logical change per commit
- conventional prefixes: feat:, fix:, refactor:, docs:, test:, chore:
- commit by default after completing a change

## license

cyber license: don't trust. don't fear. don't beg.
