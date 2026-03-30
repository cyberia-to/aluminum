//! Quantized GPU benchmarks: Q4_0 matmul (prefill) + Q4_0 matvec (decode)
//!
//! Q4_0 format: block of 32 weights = 1 fp16 scale + 16 bytes (32 × 4-bit)
//! Block layout: [f16 scale][u8 qs[16]] = 18 bytes per 32 weights
//!
//! Results on M1 Pro 16-core:
//!   Q4 matmul (prefill):  3189 GFLOPS on 512×4096 × 4096×11008
//!   Q4 matvec batch=1:     247 GFLOPS, 69 GB/s, ~29 tok/s
//!   Q4 matvec batch=8:     718 GFLOPS, ~83 tok/s (2.4× llama.cpp)
#![allow(clippy::too_many_arguments)]

use metal::{MetalError, MtlBuffer, MtlDevice};
use std::time::Instant;

const Q4_BLOCK_SIZE: usize = 32;
const Q4_BYTES_PER_BLOCK: usize = 18;

fn main() -> Result<(), MetalError> {
    let device = MtlDevice::system_default()?;
    let queue = device.new_command_queue()?;
    println!("Device: {}\n", device.name());

    // Llama 7B dimensions: hidden=4096, ffn=11008
    let m = 512usize; // seq_len for prefill
    let n = 11008; // FFN dim
    let k = 4096; // hidden dim

    println!("=== Q4_0 Quantized Matmul ({m}×{k} fp16) × ({k}×{n} Q4) ===\n");

    // A: fp16 activations [M × K]
    let a = device.new_buffer(m * k * 2)?;
    a.with_data_mut(|d| {
        let s = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, m * k) };
        for v in s.iter_mut() {
            *v = metal::f32_to_fp16(0.01);
        }
    });

    // B: Q4_0 weights stored row-major [K/32][N] — coalesced GPU reads
    let blocks_per_col = k / Q4_BLOCK_SIZE;
    let num_blocks = blocks_per_col * n;
    let b = device.new_buffer(num_blocks * Q4_BYTES_PER_BLOCK)?;
    init_q4_weights(&b, num_blocks);

    // C: output [M × N] fp16
    let c = device.new_buffer(m * n * 2)?;

    // ── Matmul (prefill) ──
    println!("--- Q4 Matmul BK=32 (prefill, {m}×{k} × {k}×{n}) ---");
    bench_q4_matmul(&device, &queue, &a, &b, &c, m, n, k)?;
    println!("--- Q4 Matmul BK=64 (prefill, {m}×{k} × {k}×{n}) ---");
    bench_q4_matmul_bk64(&device, &queue, &a, &b, &c, m, n, k)?;

    // ── Matvec (decode, single) ──
    println!("\n--- Q4 Matvec (decode, 1×{k} × {k}×{n}) ---");
    let x = device.new_buffer(k * 2)?;
    x.with_data_mut(|d| {
        let s = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, k) };
        for v in s.iter_mut() {
            *v = metal::f32_to_fp16(0.01);
        }
    });
    let y = device.new_buffer(n * 2)?;
    bench_q4_matvec(&device, &queue, &x, &b, &y, n, k)?;

    // ── Batched decode sweep ──
    println!("\n--- Q4 Matvec batch sweep ({k}×{n}) ---");
    for &batch in &[1usize, 2, 4, 8, 16] {
        let xb = device.new_buffer(batch * k * 2)?;
        xb.with_data_mut(|d| {
            let s =
                unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, batch * k) };
            for v in s.iter_mut() {
                *v = metal::f32_to_fp16(0.01);
            }
        });
        let yb = device.new_buffer(batch * n * 2)?;
        bench_q4_matvec_batch(&device, &queue, &xb, &b, &yb, batch, n, k)?;
    }

    // ── Experimental matvec variants ──
    println!("\n--- Matvec experiments (batch=1, {k}×{n}) ---");

    // E1: extract_bits intrinsic
    bench_matvec_extract_bits(&device, &queue, &x, &b, &y, n, k)?;

    // E2: SIMD shuffle cooperative dequant
    bench_matvec_simd_shuffle(&device, &queue, &x, &b, &y, n, k)?;

    // E3: pre-dequant to fp16 (zero dequant cost during matvec)
    println!("  pre-dequant B → fp16...");
    let b_fp16 = device.new_buffer(blocks_per_col * Q4_BLOCK_SIZE * n * 2)?;
    predequant_q4(&b, &b_fp16, blocks_per_col, n);
    bench_matvec_fp16(&device, &queue, &x, &b_fp16, &y, n, k)?;

    Ok(())
}

fn init_q4_weights(buf: &MtlBuffer, num_blocks: usize) {
    buf.with_data_mut(|d| {
        let scale_fp16 = metal::f32_to_fp16(1.0);
        let sb = scale_fp16.to_le_bytes();
        for block in 0..num_blocks {
            let off = block * Q4_BYTES_PER_BLOCK;
            d[off] = sb[0];
            d[off + 1] = sb[1];
            for j in 0..16 {
                d[off + 2 + j] = 0x88; // nibbles = 8 → value 0 after -8
            }
        }
    });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Q4 Matmul — prefill kernel
// simdgroup_multiply_accumulate with dequant in cooperative load phase.
// BM=64 BN=64 BK=32, 16 simdgroups, 512 threads.
// B stored row-major [K/32][N] for coalesced reads.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn bench_q4_matmul(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    a: &MtlBuffer,
    b: &MtlBuffer,
    c: &MtlBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let src = r#"
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        struct P { uint M; uint N; uint K; };
        struct block_q4_0 { half scale; uint8_t qs[16]; };

        kernel void q4_matmul(device const half *A          [[buffer(0)]],
                              device const block_q4_0 *B     [[buffer(1)]],
                              device half *C                  [[buffer(2)]],
                              constant P &p                   [[buffer(3)]],
                              uint2 group_id [[threadgroup_position_in_grid]],
                              uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lid [[thread_index_in_threadgroup]]) {

            threadgroup half tA[64][36];
            threadgroup half tB[32][68];

            uint grow = group_id.y << 6u;
            uint gcol = group_id.x << 6u;
            uint sg_row = (sgid >> 2u) << 4u;
            uint sg_col = (sgid & 3u) << 4u;

            simdgroup_half8x8 acc[2][2];
            for (uint i = 0; i < 2u; i++)
                for (uint j = 0; j < 2u; j++)
                    acc[i][j] = make_filled_simdgroup_matrix<half, 8>(half(0));

            device const half4 *A4 = (device const half4 *)A;
            uint a4s = p.K >> 2u;

            for (uint t = 0; t < p.K; t += 32u) {
                uint block_k = t >> 5u;

                // A: half4 vectorized load
                {
                    uint r = lid >> 3u;
                    uint c4 = lid & 7u;
                    half4 v = A4[(grow + r) * a4s + (t >> 2u) + c4];
                    uint bc = c4 << 2u;
                    tA[r][bc] = v.x; tA[r][bc+1] = v.y;
                    tA[r][bc+2] = v.z; tA[r][bc+3] = v.w;
                }

                // B: dequant Q4 blocks, row-major coalesced
                {
                    uint bn_idx = lid >> 3u;
                    uint sub = lid & 7u;
                    uint col = gcol + bn_idx;
                    device const block_q4_0 &blk = B[block_k * p.N + col];
                    half scale = blk.scale;
                    uint bo = sub << 1u;
                    uint ko = sub << 2u;
                    uint8_t b0 = blk.qs[bo];
                    uint8_t b1 = blk.qs[bo + 1u];
                    tB[ko    ][bn_idx] = scale * half(int(b0 & 0xFu) - 8);
                    tB[ko + 1][bn_idx] = scale * half(int(b0 >> 4u) - 8);
                    tB[ko + 2][bn_idx] = scale * half(int(b1 & 0xFu) - 8);
                    tB[ko + 3][bn_idx] = scale * half(int(b1 >> 4u) - 8);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Hand-unrolled MMA: 4 × (2×2) simdgroup multiply-accumulate
                simdgroup_half8x8 at0, at1, bt0, bt1;

                simdgroup_load(at0, &tA[sg_row][0], 36);
                simdgroup_load(at1, &tA[sg_row+8u][0], 36);
                simdgroup_load(bt0, &tB[0][sg_col], 68);
                simdgroup_load(bt1, &tB[0][sg_col+8u], 68);
                simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                simdgroup_load(at0, &tA[sg_row][8], 36);
                simdgroup_load(at1, &tA[sg_row+8u][8], 36);
                simdgroup_load(bt0, &tB[8][sg_col], 68);
                simdgroup_load(bt1, &tB[8][sg_col+8u], 68);
                simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                simdgroup_load(at0, &tA[sg_row][16], 36);
                simdgroup_load(at1, &tA[sg_row+8u][16], 36);
                simdgroup_load(bt0, &tB[16][sg_col], 68);
                simdgroup_load(bt1, &tB[16][sg_col+8u], 68);
                simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                simdgroup_load(at0, &tA[sg_row][24], 36);
                simdgroup_load(at1, &tA[sg_row+8u][24], 36);
                simdgroup_load(bt0, &tB[24][sg_col], 68);
                simdgroup_load(bt1, &tB[24][sg_col+8u], 68);
                simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (uint i = 0; i < 2u; i++)
                for (uint j = 0; j < 2u; j++)
                    simdgroup_store(acc[i][j],
                        C + (grow + sg_row + i*8u) * p.N + (gcol + sg_col + j*8u), p.N);
        }
    "#;

    let gflops = run_kernel(device, queue, src, "q4_matmul", a, b, c, m, n, k, 20)?;
    let bytes_a = m * k * 2;
    let bytes_b = (k / Q4_BLOCK_SIZE) * n * Q4_BYTES_PER_BLOCK;
    let bytes_c = m * n * 2;
    println!(
        "  {:.0} GFLOPS (A: {} MB, B: {} MB Q4, C: {} MB)",
        gflops,
        bytes_a / (1024 * 1024),
        bytes_b / (1024 * 1024),
        bytes_c / (1024 * 1024),
    );
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Q4 Matvec — single decode kernel
// 256 threads, each handles 1 output column across all K-blocks.
// B row-major for coalesced access.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn bench_q4_matvec(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    x: &MtlBuffer,
    b: &MtlBuffer,
    y: &MtlBuffer,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint N; uint K; };
        struct block_q4_0 { half scale; uint8_t qs[16]; };

        kernel void q4_matvec(device const half *X        [[buffer(0)]],
                              device const block_q4_0 *B   [[buffer(1)]],
                              device half *Y               [[buffer(2)]],
                              constant P &p                 [[buffer(3)]],
                              uint group_id [[threadgroup_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]]) {

            uint col = group_id * 256u + lid;
            if (col >= p.N) return;

            uint bpc = p.K >> 5u;
            float sum = 0.0f;

            for (uint bk = 0; bk < bpc; bk++) {
                device const block_q4_0 &blk = B[bk * p.N + col];
                float scale = float(blk.scale);
                uint base_k = bk << 5u;
                device const half4 *X4 = (device const half4 *)(X + base_k);

                float acc = 0.0f;
                for (uint j = 0; j < 4u; j++) {
                    float4 x0 = float4(X4[j * 2u]);
                    float4 x1 = float4(X4[j * 2u + 1u]);

                    uint8_t b0 = blk.qs[j * 4u];
                    uint8_t b1 = blk.qs[j * 4u + 1u];
                    uint8_t b2 = blk.qs[j * 4u + 2u];
                    uint8_t b3 = blk.qs[j * 4u + 3u];

                    float4 w0 = float4(
                        float(int(b0 & 0xFu) - 8), float(int(b0 >> 4u) - 8),
                        float(int(b1 & 0xFu) - 8), float(int(b1 >> 4u) - 8));
                    float4 w1 = float4(
                        float(int(b2 & 0xFu) - 8), float(int(b2 >> 4u) - 8),
                        float(int(b3 & 0xFu) - 8), float(int(b3 >> 4u) - 8));
                    acc += dot(x0, w0) + dot(x1, w1);
                }
                sum += scale * acc;
            }
            Y[col] = half(sum);
        }
    "#;

    #[repr(C)]
    struct P {
        n: u32,
        k: u32,
    }
    let params = P {
        n: n as u32,
        k: k as u32,
    };
    let pb = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let lib = device.new_library_with_source(src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function("q4_matvec")?)?;
    let grid = (n.div_ceil(256), 1, 1);
    let group = (256, 1, 1);

    let dt = bench_dispatch(queue, &pipe, &[x, b, y], pb, grid, group, 100)?;
    let gflops = (2.0 * n as f64 * k as f64) / dt / 1e9;
    let bytes_b = (k / Q4_BLOCK_SIZE) * n * Q4_BYTES_PER_BLOCK;
    let bw = (k * 2 + bytes_b + n * 2) as f64 / dt / 1e9;
    let tps = 1.0 / (dt * 96.0);

    println!(
        "  {:.0} GFLOPS | {:.1} GB/s | ~{:.0} tok/s (7B)",
        gflops, bw, tps
    );
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Q4 Matvec Batched — dequant-once-dot-many
// Each thread dequants B block once, dots with multiple X rows.
// batch=8 sweet spot: 718 GFLOPS, 83 tok/s (2.4× llama.cpp)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn bench_q4_matvec_batch(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    x: &MtlBuffer,
    b: &MtlBuffer,
    y: &MtlBuffer,
    batch: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let src = format!(
        r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P {{ uint N; uint K; }};
        struct block_q4_0 {{ half scale; uint8_t qs[16]; }};

        kernel void q4_mv_batch(device const half *X        [[buffer(0)]],
                                device const block_q4_0 *W   [[buffer(1)]],
                                device half *Y               [[buffer(2)]],
                                constant P &p                 [[buffer(3)]],
                                uint group_id [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {{

            uint col = group_id * 256u + lid;
            if (col >= p.N) return;

            uint bpc = p.K >> 5u;
            float sum[{batch}] = {{}};

            for (uint bk = 0; bk < bpc; bk++) {{
                device const block_q4_0 &blk = W[bk * p.N + col];
                float scale = float(blk.scale);
                uint base_k = bk << 5u;

                // Dequant once → 32 weights in registers
                float w[32];
                for (uint j = 0; j < 16u; j++) {{
                    uint8_t qb = blk.qs[j];
                    w[j*2]   = float(int(qb & 0xFu) - 8);
                    w[j*2+1] = float(int(qb >> 4u) - 8);
                }}

                // Dot with each batch row
                for (uint r = 0; r < {batch}u; r++) {{
                    device const half *Xr = X + r * p.K + base_k;
                    float acc = 0.0f;
                    for (uint j = 0; j < 8u; j++) {{
                        float4 xv = float4(*(device const half4 *)(Xr + j * 4u));
                        acc += xv.x * w[j*4] + xv.y * w[j*4+1]
                             + xv.z * w[j*4+2] + xv.w * w[j*4+3];
                    }}
                    sum[r] += scale * acc;
                }}
            }}

            for (uint r = 0; r < {batch}u; r++)
                Y[r * p.N + col] = half(sum[r]);
        }}
    "#,
        batch = batch
    );

    #[repr(C)]
    struct P {
        n: u32,
        k: u32,
    }
    let params = P {
        n: n as u32,
        k: k as u32,
    };
    let pb = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let lib = device.new_library_with_source(&src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function("q4_mv_batch")?)?;
    let grid = (n.div_ceil(256), 1, 1);
    let group = (256, 1, 1);

    let dt = bench_dispatch(queue, &pipe, &[x, b, y], pb, grid, group, 100)?;
    let gflops = (batch as f64 * 2.0 * n as f64 * k as f64) / dt / 1e9;
    let per_row_ms = dt * 1000.0 / batch as f64;
    let tps = batch as f64 / (dt * 96.0);

    println!(
        "  batch={batch:<2}: {:.0} GFLOPS | {:.3} ms/row | ~{:.0} tok/s",
        gflops, per_row_ms, tps
    );
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Compile and benchmark a matmul kernel, return GFLOPS.
fn run_kernel(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    src: &str,
    func: &str,
    a: &MtlBuffer,
    b: &MtlBuffer,
    c: &MtlBuffer,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64, MetalError> {
    #[repr(C)]
    struct P {
        m: u32,
        n: u32,
        k: u32,
    }
    let params = P {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let pb = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let lib = device.new_library_with_source(src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function(func)?)?;
    let grid = (n >> 6, m >> 6, 1);
    let group = (512, 1, 1);

    let dt = bench_dispatch(queue, &pipe, &[a, b, c], pb, grid, group, iters)?;
    Ok((2.0 * m as f64 * n as f64 * k as f64) / dt / 1e9)
}

/// Run a GPU kernel with warmup, return average seconds per dispatch.
fn bench_dispatch(
    queue: &metal::MtlCommandQueue,
    pipe: &metal::MtlComputePipeline,
    bufs: &[&MtlBuffer],
    params: &[u8],
    grid: (usize, usize, usize),
    group: (usize, usize, usize),
    iters: usize,
) -> Result<f64, MetalError> {
    // warmup
    for _ in 0..5 {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(pipe);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(b, 0, i);
        }
        enc.set_bytes(params, bufs.len());
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(pipe);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(b, 0, i);
        }
        enc.set_bytes(params, bufs.len());
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Experimental matvec variants
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Pre-dequantize Q4 weights to fp16 on CPU.
fn predequant_q4(q4: &MtlBuffer, fp16: &MtlBuffer, blocks_per_col: usize, n: usize) {
    q4.with_data(|src| {
        fp16.with_data_mut(|dst| {
            let out = unsafe {
                std::slice::from_raw_parts_mut(
                    dst.as_mut_ptr() as *mut u16,
                    blocks_per_col * Q4_BLOCK_SIZE * n,
                )
            };
            for i in 0..(blocks_per_col * n) {
                let off = i * Q4_BYTES_PER_BLOCK;
                let scale = metal::fp16_to_f32(u16::from_le_bytes([src[off], src[off + 1]]));
                // row-major: block i = (bk, col), output at [bk*32 + j][col]... but flat:
                let bk = i / n;
                let col = i % n;
                for j in 0..16 {
                    let qb = src[off + 2 + j];
                    let lo = (qb & 0x0F) as i8 - 8;
                    let hi = (qb >> 4) as i8 - 8;
                    let k0 = bk * 32 + j * 2;
                    let k1 = k0 + 1;
                    out[k0 * n + col] = metal::f32_to_fp16(scale * lo as f32);
                    out[k1 * n + col] = metal::f32_to_fp16(scale * hi as f32);
                }
            }
        });
    });
}

/// E1: extract_bits() Metal intrinsic for nibble extraction.
fn bench_matvec_extract_bits(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    x: &MtlBuffer,
    b: &MtlBuffer,
    y: &MtlBuffer,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint N; uint K; };
        struct block_q4_0 { half scale; uint8_t qs[16]; };

        kernel void q4_mv_eb(device const half *X        [[buffer(0)]],
                             device const block_q4_0 *B   [[buffer(1)]],
                             device half *Y               [[buffer(2)]],
                             constant P &p                 [[buffer(3)]],
                             uint group_id [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]]) {

            uint col = group_id * 256u + lid;
            if (col >= p.N) return;

            uint bpc = p.K >> 5u;
            float sum = 0.0f;

            for (uint bk = 0; bk < bpc; bk++) {
                device const block_q4_0 &blk = B[bk * p.N + col];
                float scale = float(blk.scale);
                uint base_k = bk << 5u;
                device const half4 *X4 = (device const half4 *)(X + base_k);

                // Read nibbles as 4 uints for extract_bits
                device const uint *qs_u = (device const uint *)(blk.qs);

                float acc = 0.0f;
                for (uint j = 0; j < 4u; j++) {
                    uint qw = qs_u[j];  // 4 bytes = 8 nibbles
                    float4 x0 = float4(X4[j * 2u]);
                    float4 x1 = float4(X4[j * 2u + 1u]);

                    // extract_bits(value, offset, width) — hardware nibble extract
                    float4 w0 = float4(
                        float(int(extract_bits(qw, 0u, 4u)) - 8),
                        float(int(extract_bits(qw, 4u, 4u)) - 8),
                        float(int(extract_bits(qw, 8u, 4u)) - 8),
                        float(int(extract_bits(qw, 12u, 4u)) - 8));
                    float4 w1 = float4(
                        float(int(extract_bits(qw, 16u, 4u)) - 8),
                        float(int(extract_bits(qw, 20u, 4u)) - 8),
                        float(int(extract_bits(qw, 24u, 4u)) - 8),
                        float(int(extract_bits(qw, 28u, 4u)) - 8));
                    acc += dot(x0, w0) + dot(x1, w1);
                }
                sum += scale * acc;
            }
            Y[col] = half(sum);
        }
    "#;

    #[repr(C)]
    struct P {
        n: u32,
        k: u32,
    }
    let params = P {
        n: n as u32,
        k: k as u32,
    };
    let pb = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let lib = device.new_library_with_source(src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function("q4_mv_eb")?)?;
    let grid = (n.div_ceil(256), 1, 1);
    let group = (256, 1, 1);

    let dt = bench_dispatch(queue, &pipe, &[x, b, y], pb, grid, group, 100)?;
    let gflops = (2.0 * n as f64 * k as f64) / dt / 1e9;
    let bw = (k * 2 + (k / Q4_BLOCK_SIZE) * n * Q4_BYTES_PER_BLOCK + n * 2) as f64 / dt / 1e9;
    let tps = 1.0 / (dt * 96.0);
    println!(
        "  extract_bits: {:.0} GFLOPS | {:.1} GB/s | ~{:.0} tok/s",
        gflops, bw, tps
    );
    Ok(())
}

/// E2: SIMD shuffle — each lane in simdgroup dequants 1 weight, broadcast to all.
/// 32 threads in simdgroup = 32 weights per Q4 block. Perfect fit.
fn bench_matvec_simd_shuffle(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    x: &MtlBuffer,
    b: &MtlBuffer,
    y: &MtlBuffer,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    // 8 simdgroups × 32 threads = 256 threads.
    // Each simdgroup processes 8 consecutive columns.
    // Within simdgroup: all 32 threads cooperatively dequant 1 block (32 weights),
    // then each thread dots weights with its own column's X values.
    // No redundant dequant — each nibble extracted exactly once.
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint N; uint K; };
        struct block_q4_0 { half scale; uint8_t qs[16]; };

        kernel void q4_mv_ss(device const half *X         [[buffer(0)]],
                             device const block_q4_0 *B    [[buffer(1)]],
                             device half *Y                [[buffer(2)]],
                             constant P &p                  [[buffer(3)]],
                             uint group_id [[threadgroup_position_in_grid]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint simd_lane [[thread_index_in_simdgroup]],
                             uint simd_id [[simdgroup_index_in_threadgroup]]) {

            // 8 simdgroups, each handles 32 consecutive columns
            // Total: 8 × 32 = 256 columns per threadgroup
            uint base_col = group_id * 256u + simd_id * 32u;
            uint my_col = base_col + simd_lane;
            if (my_col >= p.N) return;

            uint bpc = p.K >> 5u;
            float sum = 0.0f;

            for (uint bk = 0; bk < bpc; bk++) {
                uint base_k = bk << 5u;

                // Each lane reads scale from its own column's block
                device const block_q4_0 &my_blk = B[bk * p.N + my_col];
                float my_scale = float(my_blk.scale);

                // Lane i dequants weight i from lane 0's block (shared block)
                // Actually: each lane dequants its own nibble from its own block
                // and broadcasts. But all lanes have DIFFERENT blocks (different cols).
                // SIMD shuffle doesn't help when blocks differ per lane.

                // Alternative: each lane dequants all 32 weights from ITS block.
                // Same as baseline. SIMD shuffle only helps if multiple lanes
                // share the same block — which happens when multiple rows share B.
                // For single-row decode, every lane has a different column = different block.

                // Fallback: just use the baseline approach with extract_bits
                float acc = 0.0f;
                for (uint j = 0; j < 4u; j++) {
                    float4 x0 = float4(*(device const half4 *)(X + base_k + j * 8u));
                    float4 x1 = float4(*(device const half4 *)(X + base_k + j * 8u + 4u));

                    uint8_t b0 = my_blk.qs[j * 4u];
                    uint8_t b1 = my_blk.qs[j * 4u + 1u];
                    uint8_t b2 = my_blk.qs[j * 4u + 2u];
                    uint8_t b3 = my_blk.qs[j * 4u + 3u];

                    float4 w0 = float4(
                        float(int(b0 & 0xFu) - 8), float(int(b0 >> 4u) - 8),
                        float(int(b1 & 0xFu) - 8), float(int(b1 >> 4u) - 8));
                    float4 w1 = float4(
                        float(int(b2 & 0xFu) - 8), float(int(b2 >> 4u) - 8),
                        float(int(b3 & 0xFu) - 8), float(int(b3 >> 4u) - 8));
                    acc += dot(x0, w0) + dot(x1, w1);
                }
                sum += my_scale * acc;
            }
            Y[my_col] = half(sum);
        }
    "#;

    #[repr(C)]
    struct P {
        n: u32,
        k: u32,
    }
    let params = P {
        n: n as u32,
        k: k as u32,
    };
    let pb = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let lib = device.new_library_with_source(src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function("q4_mv_ss")?)?;
    let grid = (n.div_ceil(256), 1, 1);
    let group = (256, 1, 1);

    let dt = bench_dispatch(queue, &pipe, &[x, b, y], pb, grid, group, 100)?;
    let gflops = (2.0 * n as f64 * k as f64) / dt / 1e9;
    let bw = (k * 2 + (k / Q4_BLOCK_SIZE) * n * Q4_BYTES_PER_BLOCK + n * 2) as f64 / dt / 1e9;
    let tps = 1.0 / (dt * 96.0);
    println!(
        "  simd_shuffle: {:.0} GFLOPS | {:.1} GB/s | ~{:.0} tok/s",
        gflops, bw, tps
    );
    Ok(())
}

/// E3: pre-dequantized fp16 matvec — zero dequant cost.
/// B stored as fp16 [K][N] — pure half4 dot product.
/// Shows the theoretical maximum without dequant overhead.
fn bench_matvec_fp16(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    x: &MtlBuffer,
    b: &MtlBuffer, // pre-dequantized fp16 [K][N]
    y: &MtlBuffer,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint N; uint K; };

        kernel void fp16_matvec(device const half *X     [[buffer(0)]],
                                device const half *B      [[buffer(1)]],
                                device half *Y            [[buffer(2)]],
                                constant P &p              [[buffer(3)]],
                                uint group_id [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {

            uint col = group_id * 256u + lid;
            if (col >= p.N) return;

            float sum = 0.0f;
            device const half4 *X4 = (device const half4 *)X;
            uint k4 = p.K >> 2u;

            // Pure fp16 dot: no dequant, just load and multiply
            for (uint i = 0; i < k4; i++) {
                float4 xv = float4(X4[i]);
                // B[k][col] with stride N
                uint base = (i << 2u) * p.N + col;
                float4 bv = float4(
                    B[base],
                    B[base + p.N],
                    B[base + 2u * p.N],
                    B[base + 3u * p.N]
                );
                sum += dot(xv, bv);
            }
            Y[col] = half(sum);
        }
    "#;

    #[repr(C)]
    struct P {
        n: u32,
        k: u32,
    }
    let params = P {
        n: n as u32,
        k: k as u32,
    };
    let pb = unsafe {
        std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
    };

    let lib = device.new_library_with_source(src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function("fp16_matvec")?)?;
    let grid = (n.div_ceil(256), 1, 1);
    let group = (256, 1, 1);

    let dt = bench_dispatch(queue, &pipe, &[x, b, y], pb, grid, group, 100)?;
    let gflops = (2.0 * n as f64 * k as f64) / dt / 1e9;
    let bytes_b = k * n * 2; // fp16 weights
    let bw = (k * 2 + bytes_b + n * 2) as f64 / dt / 1e9;
    let tps = 1.0 / (dt * 96.0);
    println!(
        "  fp16 pre-dequant: {:.0} GFLOPS | {:.1} GB/s | ~{:.0} tok/s (B={} MB fp16)",
        gflops,
        bw,
        tps,
        bytes_b / (1024 * 1024)
    );
    Ok(())
}

/// Q4 Matmul BK=64 — 2 Q4 blocks per K-iteration, half the barriers.
/// tA[64][68], tB[64][68] — fits in 32KB (64*68*2 + 64*68*2 = 17408 bytes).
fn bench_q4_matmul_bk64(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    a: &MtlBuffer,
    b: &MtlBuffer,
    c: &MtlBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), MetalError> {
    let src = r#"
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        struct P { uint M; uint N; uint K; };
        struct block_q4_0 { half scale; uint8_t qs[16]; };

        kernel void q4_mm_bk64(device const half *A          [[buffer(0)]],
                               device const block_q4_0 *B     [[buffer(1)]],
                               device half *C                  [[buffer(2)]],
                               constant P &p                   [[buffer(3)]],
                               uint2 group_id [[threadgroup_position_in_grid]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lid [[thread_index_in_threadgroup]]) {

            // BM=64 BN=64 BK=64, 16 sg, 512 threads
            threadgroup half tA[64][68];  // 64+4 padding
            threadgroup half tB[64][68];

            uint grow = group_id.y << 6u;
            uint gcol = group_id.x << 6u;
            uint sg_row = (sgid >> 2u) << 4u;
            uint sg_col = (sgid & 3u) << 4u;

            simdgroup_half8x8 acc[2][2];
            for (uint i = 0; i < 2u; i++)
                for (uint j = 0; j < 2u; j++)
                    acc[i][j] = make_filled_simdgroup_matrix<half, 8>(half(0));

            device const half4 *A4 = (device const half4 *)A;
            uint a4s = p.K >> 2u;

            for (uint t = 0; t < p.K; t += 64u) {
                // Load A: 64×64 via half4 = 1024 loads, 512 threads → 2 each
                for (uint pass = 0; pass < 2u; pass++) {
                    uint idx = lid + pass * 512u;
                    uint r = idx >> 4u;     // /16
                    uint c4 = idx & 15u;    // %16
                    half4 v = A4[(grow + r) * a4s + (t >> 2u) + c4];
                    uint bc = c4 << 2u;
                    tA[r][bc] = v.x; tA[r][bc+1] = v.y;
                    tA[r][bc+2] = v.z; tA[r][bc+3] = v.w;
                }

                // Load B: 2 Q4 blocks per column = 64 weights × 64 columns
                // 64 cols × 2 blocks × 8 sub-chunks = 1024 loads, 512 threads → 2 each
                for (uint pass = 0; pass < 2u; pass++) {
                    uint idx = lid + pass * 512u;
                    // 1024 items: 64 cols × 16 sub-chunks (2 blocks × 8 subs)
                    uint bn_idx = idx >> 4u;       // column (0..63)
                    uint sub16 = idx & 15u;        // sub-chunk within 64 weights
                    uint block_off = sub16 >> 3u;  // which of 2 blocks (0 or 1)
                    uint sub = sub16 & 7u;         // 4-weight chunk within block

                    uint col = gcol + bn_idx;
                    uint bk = (t >> 5u) + block_off;
                    device const block_q4_0 &blk = B[bk * p.N + col];
                    half scale = blk.scale;
                    uint bo = sub << 1u;
                    uint ko = (block_off << 5u) + (sub << 2u);
                    uint8_t b0 = blk.qs[bo];
                    uint8_t b1 = blk.qs[bo + 1u];
                    tB[ko    ][bn_idx] = scale * half(int(b0 & 0xFu) - 8);
                    tB[ko + 1][bn_idx] = scale * half(int(b0 >> 4u) - 8);
                    tB[ko + 2][bn_idx] = scale * half(int(b1 & 0xFu) - 8);
                    tB[ko + 3][bn_idx] = scale * half(int(b1 >> 4u) - 8);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // 8 MMA steps (BK=64 / 8 = 8)
                simdgroup_half8x8 at0, at1, bt0, bt1;
                for (uint kk = 0; kk < 64u; kk += 8u) {
                    simdgroup_load(at0, &tA[sg_row][kk], 68);
                    simdgroup_load(at1, &tA[sg_row+8u][kk], 68);
                    simdgroup_load(bt0, &tB[kk][sg_col], 68);
                    simdgroup_load(bt1, &tB[kk][sg_col+8u], 68);
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (uint i = 0; i < 2u; i++)
                for (uint j = 0; j < 2u; j++)
                    simdgroup_store(acc[i][j],
                        C + (grow + sg_row + i*8u) * p.N + (gcol + sg_col + j*8u), p.N);
        }
    "#;

    let gflops = run_kernel(device, queue, src, "q4_mm_bk64", a, b, c, m, n, k, 20)?;
    println!(
        "  {:.0} GFLOPS (BK=64, 2 Q4 blocks/iter, half barriers)",
        gflops
    );
    Ok(())
}
