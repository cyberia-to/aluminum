//! Q4_K quantization benchmark — llama.cpp compatible format
//!
//! Q4_K: 256-element super-blocks with 6-bit sub-block scales.
//! struct block_q4_K {
//!     half d;            // super-block scale (2 bytes)
//!     half dmin;         // super-block minimum (2 bytes)
//!     uint8_t scales[12]; // 16 sub-block scales, 6-bit packed (12 bytes)
//!     uint8_t qs[128];   // 256 × 4-bit nibbles (128 bytes)
//! } // = 144 bytes per 256 weights = 4.5 bits/weight
#![allow(clippy::too_many_arguments)]

use metal::{MetalError, MtlBuffer, MtlDevice};
use std::time::Instant;

const Q4K_BLOCK_SIZE: usize = 256;
const Q4K_BYTES_PER_BLOCK: usize = 144;

fn main() -> Result<(), MetalError> {
    let device = MtlDevice::system_default()?;
    let queue = device.new_command_queue()?;
    println!("Device: {}\n", device.name());

    let n = 11008usize; // Llama 7B FFN
    let k = 4096; // hidden

    // B: Q4_K weights, row-major [K/256][N]
    let blocks_per_col = k / Q4K_BLOCK_SIZE; // 16
    let num_blocks = blocks_per_col * n;
    let b = device.new_buffer(num_blocks * Q4K_BYTES_PER_BLOCK)?;
    init_q4k_weights(&b, num_blocks);

    // X: fp16 input [K]
    let x = device.new_buffer(k * 2)?;
    x.with_data_mut(|d| {
        let s = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, k) };
        for v in s.iter_mut() {
            *v = metal::f32_to_fp16(0.01);
        }
    });
    let y = device.new_buffer(n * 2)?;

    println!("=== Q4_K Matvec (decode, 1×{k} × {k}×{n}) ===");
    println!(
        "  B: {:.1} MB ({} blocks × {} bytes)\n",
        (num_blocks * Q4K_BYTES_PER_BLOCK) as f64 / (1024.0 * 1024.0),
        num_blocks,
        Q4K_BYTES_PER_BLOCK
    );

    bench_q4k_matvec(&device, &queue, &x, &b, &y, n, k)?;

    // Batched
    println!("\n--- Q4_K batch sweep ---");
    for &batch in &[1usize, 2, 4, 8] {
        let xb = device.new_buffer(batch * k * 2)?;
        xb.with_data_mut(|d| {
            let s =
                unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, batch * k) };
            for v in s.iter_mut() {
                *v = metal::f32_to_fp16(0.01);
            }
        });
        let yb = device.new_buffer(batch * n * 2)?;
        bench_q4k_matvec_batch(&device, &queue, &xb, &b, &yb, batch, n, k)?;
    }

    Ok(())
}

fn init_q4k_weights(buf: &MtlBuffer, num_blocks: usize) {
    buf.with_data_mut(|d| {
        for block in 0..num_blocks {
            let off = block * Q4K_BYTES_PER_BLOCK;
            // d (scale) = fp16(1.0)
            let s = metal::f32_to_fp16(1.0).to_le_bytes();
            d[off] = s[0];
            d[off + 1] = s[1];
            // dmin = fp16(0.0)
            d[off + 2] = 0;
            d[off + 3] = 0;
            // scales[12]: all 32 (mid-range 6-bit value)
            for j in 0..12 {
                d[off + 4 + j] = 0x20;
            }
            // qs[128]: all 0x88 (nibble=8, centered)
            for j in 0..128 {
                d[off + 16 + j] = 0x88;
            }
        }
    });
}

/// Q4_K matvec: each thread handles 1 output column
fn bench_q4k_matvec(
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

        // Q4_K: 256-element super-block
        struct block_q4_K {
            half d;            // super-block scale
            half dmin;         // super-block min
            uint8_t scales[12]; // 6-bit sub-block scales (packed)
            uint8_t qs[128];   // 256 × 4-bit
        };

        kernel void q4k_matvec(device const half *X            [[buffer(0)]],
                               device const block_q4_K *B       [[buffer(1)]],
                               device half *Y                    [[buffer(2)]],
                               constant P &p                     [[buffer(3)]],
                               uint group_id [[threadgroup_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]]) {

            uint col = group_id * 256u + lid;
            if (col >= p.N) return;

            uint bpc = p.K >> 8u;  // blocks per col = K/256
            float sum = 0.0f;

            for (uint bk = 0; bk < bpc; bk++) {
                device const block_q4_K &blk = B[bk * p.N + col];
                float d = float(blk.d);
                float dmin = float(blk.dmin);
                uint base_k = bk << 8u; // bk * 256

                // Decode 6-bit scales from packed 12 bytes → 16 scales
                // scales[0..5] contain low 6 bits of scales 0..7
                // scales[6..11] contain low 6 bits of scales 8..15
                // High 2 bits are packed differently in Q4_K
                // Simplified: treat as 8×(4-bit scale, 4-bit min) for now
                // This is approximate — real Q4_K has complex scale packing

                float acc = 0.0f;
                // 16 sub-blocks of 16 weights each
                for (uint sb = 0; sb < 16u; sb++) {
                    // Get sub-block scale (simplified: use raw byte)
                    uint8_t sc_byte = blk.scales[sb < 8u ? sb : sb - 8u + 4u];
                    float sc = float(sc_byte & 0x3Fu);  // 6-bit scale
                    float mn = float((sc_byte >> 4u) & 0x3u); // 2-bit min (simplified)

                    float sub_acc = 0.0f;
                    uint qs_off = sb * 8u;  // 16 nibbles = 8 bytes

                    for (uint j = 0; j < 8u; j++) {
                        uint8_t qb = blk.qs[qs_off + j];
                        uint k_idx = base_k + sb * 16u + j * 2u;

                        float x0 = float(X[k_idx]);
                        float x1 = float(X[k_idx + 1u]);
                        float w0 = float(int(qb & 0xFu));
                        float w1 = float(int(qb >> 4u));

                        sub_acc += x0 * w0 + x1 * w1;
                    }
                    acc += d * sc * sub_acc - dmin * mn * 16.0f;
                }
                sum += acc;
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
    let pipe = device.new_compute_pipeline(&lib.get_function("q4k_matvec")?)?;
    let grid = (n.div_ceil(256), 1, 1);
    let group = (256, 1, 1);

    // warmup
    for _ in 0..10 {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipe);
        enc.set_buffer(x, 0, 0);
        enc.set_buffer(b, 0, 1);
        enc.set_buffer(y, 0, 2);
        enc.set_bytes(pb, 3);
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipe);
        enc.set_buffer(x, 0, 0);
        enc.set_buffer(b, 0, 1);
        enc.set_buffer(y, 0, 2);
        enc.set_bytes(pb, 3);
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let gflops = (2.0 * n as f64 * k as f64) / dt / 1e9;
    let bytes_b = (k / Q4K_BLOCK_SIZE) * n * Q4K_BYTES_PER_BLOCK;
    let bw = (k * 2 + bytes_b + n * 2) as f64 / dt / 1e9;
    let tps = 1.0 / (dt * 96.0);

    println!(
        "  Q4_K single: {:.0} GFLOPS | {:.1} GB/s | ~{:.0} tok/s (7B)",
        gflops, bw, tps
    );
    Ok(())
}

/// Q4_K batched matvec
fn bench_q4k_matvec_batch(
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
        struct block_q4_K {{
            half d; half dmin;
            uint8_t scales[12];
            uint8_t qs[128];
        }};

        kernel void q4k_mv_batch(device const half *X        [[buffer(0)]],
                                 device const block_q4_K *W   [[buffer(1)]],
                                 device half *Y               [[buffer(2)]],
                                 constant P &p                 [[buffer(3)]],
                                 uint group_id [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]]) {{

            uint col = group_id * 256u + lid;
            if (col >= p.N) return;

            uint bpc = p.K >> 8u;
            float sum[{batch}] = {{}};

            for (uint bk = 0; bk < bpc; bk++) {{
                device const block_q4_K &blk = W[bk * p.N + col];
                float d = float(blk.d);
                float dmin = float(blk.dmin);
                uint base_k = bk << 8u;

                // Dequant once: 256 weights
                float w[256];
                for (uint sb = 0; sb < 16u; sb++) {{
                    uint8_t sc_byte = blk.scales[sb < 8u ? sb : sb - 8u + 4u];
                    float sc = d * float(sc_byte & 0x3Fu);
                    for (uint j = 0; j < 8u; j++) {{
                        uint8_t qb = blk.qs[sb * 8u + j];
                        w[sb*16u + j*2u]     = sc * float(int(qb & 0xFu));
                        w[sb*16u + j*2u + 1u] = sc * float(int(qb >> 4u));
                    }}
                }}

                // Dot with each batch row
                for (uint r = 0; r < {batch}u; r++) {{
                    device const half *Xr = X + r * p.K + base_k;
                    float acc = 0.0f;
                    for (uint j = 0; j < 64u; j++) {{
                        float4 xv = float4(*(device const half4 *)(Xr + j * 4u));
                        acc += xv.x * w[j*4u] + xv.y * w[j*4u+1u]
                             + xv.z * w[j*4u+2u] + xv.w * w[j*4u+3u];
                    }}
                    sum[r] += acc;
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
    let pipe = device.new_compute_pipeline(&lib.get_function("q4k_mv_batch")?)?;
    let grid = (n.div_ceil(256), 1, 1);
    let group = (256, 1, 1);

    for _ in 0..10 {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipe);
        enc.set_buffer(x, 0, 0);
        enc.set_buffer(b, 0, 1);
        enc.set_buffer(y, 0, 2);
        enc.set_bytes(pb, 3);
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipe);
        enc.set_buffer(x, 0, 0);
        enc.set_buffer(b, 0, 1);
        enc.set_buffer(y, 0, 2);
        enc.set_bytes(pb, 3);
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let gflops = (batch as f64 * 2.0 * n as f64 * k as f64) / dt / 1e9;
    let per_row_ms = dt * 1000.0 / batch as f64;
    let tps = batch as f64 / (dt * 96.0);

    println!(
        "  batch={batch:<2}: {:.0} GFLOPS | {:.3} ms/row | ~{:.0} tok/s",
        gflops, per_row_ms, tps
    );
    Ok(())
}
