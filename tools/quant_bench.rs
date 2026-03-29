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
    println!("--- Q4 Matmul (prefill, {m}×{k} × {k}×{n}) ---");
    bench_q4_matmul(&device, &queue, &a, &b, &c, m, n, k)?;

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
