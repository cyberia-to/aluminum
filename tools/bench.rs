//! GPU benchmark tool — measure buffer throughput and compute latency
#![allow(clippy::too_many_arguments)]

use metal::{MetalError, MtlDevice};
use std::time::Instant;

fn main() -> Result<(), MetalError> {
    let device = MtlDevice::system_default()?;
    println!("Device: {}", device.name());
    println!();

    // Buffer creation throughput
    let sizes = [1024, 1024 * 1024, 64 * 1024 * 1024];
    for &size in &sizes {
        let t0 = Instant::now();
        let _buf = device.new_buffer(size)?;
        let dt = t0.elapsed();
        println!(
            "Buffer {} MB: {:.2} ms",
            size / (1024 * 1024),
            dt.as_secs_f64() * 1000.0
        );
    }
    println!();

    // Compute dispatch latency
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
    let lib = device.new_library_with_source(source)?;
    let func = lib.get_function("noop")?;
    let pipeline = device.new_compute_pipeline(&func)?;
    let queue = device.new_command_queue()?;

    let n = 1024 * 1024usize;
    let buf = device.new_buffer(n * 4)?;
    buf.with_f32_mut(|d| {
        for v in d.iter_mut().take(n) {
            *v = 0.0;
        }
    });

    // Warmup
    for _ in 0..3 {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipeline);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((n, 1, 1), (256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    let iters = 100;
    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipeline);
        enc.set_buffer(&buf, 0, 0);
        enc.dispatch_threads((n, 1, 1), (256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let dt = t0.elapsed();
    let per_iter = dt.as_secs_f64() * 1000.0 / iters as f64;
    let bandwidth = (n * 4 * 2) as f64 / (per_iter / 1000.0) / 1e9;
    println!("Compute dispatch ({} floats): {:.3} ms/iter", n, per_iter);
    println!("Effective bandwidth: {:.1} GB/s", bandwidth);
    println!();

    // fp16 conversion benchmark
    let n16 = 16 * 1024 * 1024;
    let src16: Vec<u16> = (0..n16)
        .map(|i| metal::f32_to_fp16(i as f32 * 0.001))
        .collect();
    let mut dst32 = vec![0.0f32; n16];
    let mut dst16 = vec![0u16; n16];
    let src32: Vec<f32> = (0..n16).map(|i| i as f32 * 0.001).collect();

    // warmup
    metal::cvt_f16_f32(&mut dst32, &src16);
    metal::cvt_f32_f16(&mut dst16, &src32);

    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        metal::cvt_f16_f32(&mut dst32, &src16);
    }
    let dt = t0.elapsed();
    let bw = (n16 as f64 * (2 + 4) as f64 * iters as f64) / dt.as_secs_f64() / 1e9;
    println!(
        "fp16→f32 ({}M): {:.2} ms/iter, {:.1} GB/s",
        n16 / 1_000_000,
        dt.as_secs_f64() * 1000.0 / iters as f64,
        bw
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        metal::cvt_f32_f16(&mut dst16, &src32);
    }
    let dt = t0.elapsed();
    let bw = (n16 as f64 * (4 + 2) as f64 * iters as f64) / dt.as_secs_f64() / 1e9;
    println!(
        "f32→fp16 ({}M): {:.2} ms/iter, {:.1} GB/s",
        n16 / 1_000_000,
        dt.as_secs_f64() * 1000.0 / iters as f64,
        bw
    );

    // ── Matmul benchmark: naive vs tiled vs blocked vs fp16 ──
    println!();

    let naive_src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint M; uint N; uint K; };
        kernel void matmul_naive(device const float *A [[buffer(0)]],
                                 device const float *B [[buffer(1)]],
                                 device float *C       [[buffer(2)]],
                                 constant P &p [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
            uint row = gid.y, col = gid.x;
            if (row >= p.M || col >= p.N) return;
            float sum = 0.0;
            for (uint k = 0; k < p.K; k++)
                sum += A[row * p.K + k] * B[k * p.N + col];
            C[row * p.N + col] = sum;
        }
    "#;

    let tiled_src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint M; uint N; uint K; };
        constant uint TS = 32;
        kernel void matmul_tiled(device const float *A [[buffer(0)]],
                                 device const float *B [[buffer(1)]],
                                 device float *C       [[buffer(2)]],
                                 constant P &p [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]],
                                 uint2 lid [[thread_position_in_threadgroup]]) {
            threadgroup float As[32][32];
            threadgroup float Bs[32][32];
            uint row = gid.y, col = gid.x;
            uint lr = lid.y, lc = lid.x;
            float sum = 0.0;
            for (uint t = 0; t < p.K; t += TS) {
                As[lr][lc] = (row < p.M && t + lc < p.K) ? A[row * p.K + t + lc] : 0.0;
                Bs[lr][lc] = (t + lr < p.K && col < p.N) ? B[(t + lr) * p.N + col] : 0.0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 0; k < TS; k++)
                    sum += As[lr][k] * Bs[k][lc];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (row < p.M && col < p.N)
                C[row * p.N + col] = sum;
        }
    "#;

    // Register-blocked: 32x32 tile, 4x1 per thread (4 rows, 1 col)
    // 256 threads (8x32), +1 padding to avoid bank conflicts
    let blocked_src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint M; uint N; uint K; };
        constant uint TS = 32;

        kernel void matmul_blocked(device const float *A [[buffer(0)]],
                                   device const float *B [[buffer(1)]],
                                   device float *C       [[buffer(2)]],
                                   constant P &p [[buffer(3)]],
                                   uint2 gid [[thread_position_in_grid]],
                                   uint2 lid [[thread_position_in_threadgroup]]) {
            // +1 padding avoids 32-bank conflicts on column access
            threadgroup float As[32][33];
            threadgroup float Bs[32][33];

            uint base_row = (gid.y / 32) * 32 + lid.y * 4;
            uint col = gid.x;
            float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

            for (uint t = 0; t < p.K; t += TS) {
                // Each thread loads 4 elements of A (4 rows x 1 col)
                for (uint i = 0; i < 4; i++) {
                    uint r = lid.y * 4 + i;
                    uint gr = (gid.y / 32) * 32 + r;
                    As[r][lid.x] = (gr < p.M && t + lid.x < p.K) ? A[gr * p.K + t + lid.x] : 0.0f;
                }
                Bs[lid.y][lid.x] = (t + lid.y < p.K && col < p.N) ? B[(t + lid.y) * p.N + col] : 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint k = 0; k < TS; k++) {
                    float b_val = Bs[k][lid.x];
                    acc0 += As[lid.y * 4 + 0][k] * b_val;
                    acc1 += As[lid.y * 4 + 1][k] * b_val;
                    acc2 += As[lid.y * 4 + 2][k] * b_val;
                    acc3 += As[lid.y * 4 + 3][k] * b_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (col < p.N) {
                if (base_row + 0 < p.M) C[(base_row + 0) * p.N + col] = acc0;
                if (base_row + 1 < p.M) C[(base_row + 1) * p.N + col] = acc1;
                if (base_row + 2 < p.M) C[(base_row + 2) * p.N + col] = acc2;
                if (base_row + 3 < p.M) C[(base_row + 3) * p.N + col] = acc3;
            }
        }
    "#;

    // Best config from grid search: BM=64 BN=64 TM=4 TN=4 BK=16, 256 threads
    let fp16_src = r#"
        #include <metal_stdlib>
        using namespace metal;
        struct P { uint M; uint N; uint K; };

        kernel void matmul_fp16(device const half *A [[buffer(0)]],
                                device const half *B [[buffer(1)]],
                                device half *C       [[buffer(2)]],
                                constant P &p [[buffer(3)]],
                                uint2 group_id [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {
            threadgroup half tA[64][17];  // +1 padding
            threadgroup half tB[16][65];  // +1 padding

            uint tr = lid / 16u;  // 0..15
            uint tc = lid % 16u;  // 0..15
            uint grow = group_id.y * 64u;
            uint gcol = group_id.x * 64u;

            float acc[4][4] = {};

            for (uint t = 0; t < p.K; t += 16u) {
                // Load A: 64x16 = 1024 / 256 = 4 each
                for (uint i = lid; i < 64u * 16u; i += 256u) {
                    uint r = i / 16u, c = i % 16u;
                    uint gr = grow + r, gk = t + c;
                    tA[r][c] = (gr < p.M && gk < p.K) ? A[gr * p.K + gk] : half(0);
                }
                // Load B: 16x64 = 1024 / 256 = 4 each
                for (uint i = lid; i < 16u * 64u; i += 256u) {
                    uint r = i / 64u, c = i % 64u;
                    uint gk = t + r, gc = gcol + c;
                    tB[r][c] = (gk < p.K && gc < p.N) ? B[gk * p.N + gc] : half(0);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint k = 0; k < 16u; k++) {
                    float a_reg[4], b_reg[4];
                    for (uint i = 0; i < 4u; i++)
                        a_reg[i] = float(tA[tr * 4u + i][k]);
                    for (uint j = 0; j < 4u; j++)
                        b_reg[j] = float(tB[k][tc * 4u + j]);
                    for (uint i = 0; i < 4u; i++)
                        for (uint j = 0; j < 4u; j++)
                            acc[i][j] += a_reg[i] * b_reg[j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (uint i = 0; i < 4u; i++) {
                uint gr = grow + tr * 4u + i;
                if (gr >= p.M) continue;
                for (uint j = 0; j < 4u; j++) {
                    uint gc = gcol + tc * 4u + j;
                    if (gc < p.N)
                        C[gr * p.N + gc] = half(acc[i][j]);
                }
            }
        }
    "#;

    let lib_naive = device.new_library_with_source(naive_src)?;
    let lib_tiled = device.new_library_with_source(tiled_src)?;
    let lib_blocked = device.new_library_with_source(blocked_src)?;
    let lib_fp16 = device.new_library_with_source(fp16_src)?;
    let pipe_naive = device.new_compute_pipeline(&lib_naive.get_function("matmul_naive")?)?;
    let pipe_tiled = device.new_compute_pipeline(&lib_tiled.get_function("matmul_tiled")?)?;
    let pipe_blocked = device.new_compute_pipeline(&lib_blocked.get_function("matmul_blocked")?)?;
    let pipe_fp16 = device.new_compute_pipeline(&lib_fp16.get_function("matmul_fp16")?)?;

    #[repr(C)]
    struct P {
        m: u32,
        n: u32,
        k: u32,
    }

    for &dim in &[512usize, 1024, 2048] {
        let m = dim;
        let n = dim;
        let k = dim;
        let sz = m * k * 4;

        // Shared buffers
        let a = device.new_buffer(sz)?;
        let b = device.new_buffer(k * n * 4)?;
        let c = device.new_buffer(m * n * 4)?;
        a.with_f32_mut(|d| {
            for v in d.iter_mut() {
                *v = 0.01;
            }
        });
        b.with_f32_mut(|d| {
            for v in d.iter_mut() {
                *v = 0.01;
            }
        });

        // Private buffers
        let a_priv = device.new_buffer_private(sz)?;
        let b_priv = device.new_buffer_private(k * n * 4)?;
        let _c_priv = device.new_buffer_private(m * n * 4)?;

        // Copy shared -> private via blit
        {
            let cmd = queue.command_buffer()?;
            let blit = cmd.blit_encoder()?;
            blit.copy_buffer(&a, 0, &a_priv, 0, sz);
            blit.copy_buffer(&b, 0, &b_priv, 0, k * n * 4);
            blit.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let params = P {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let pb = unsafe {
            std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
        };

        let iters = if dim <= 512 {
            50
        } else if dim <= 1024 {
            20
        } else {
            5
        };

        // Helper: dispatch with threadgroups
        let bench_mm = |pipe: &metal::MtlComputePipeline,
                        ba: &metal::MtlBuffer,
                        bb: &metal::MtlBuffer,
                        bc: &metal::MtlBuffer,
                        grid: (usize, usize, usize),
                        group: (usize, usize, usize),
                        iters: usize|
         -> f64 {
            for _ in 0..3 {
                let cmd = queue.command_buffer().unwrap();
                let enc = cmd.compute_encoder().unwrap();
                enc.set_pipeline(pipe);
                enc.set_buffer(ba, 0, 0);
                enc.set_buffer(bb, 0, 1);
                enc.set_buffer(bc, 0, 2);
                enc.set_bytes(pb, 3);
                enc.dispatch_threadgroups(grid, group);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                let cmd = queue.command_buffer().unwrap();
                let enc = cmd.compute_encoder().unwrap();
                enc.set_pipeline(pipe);
                enc.set_buffer(ba, 0, 0);
                enc.set_buffer(bb, 0, 1);
                enc.set_buffer(bc, 0, 2);
                enc.set_bytes(pb, 3);
                enc.dispatch_threadgroups(grid, group);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            t0.elapsed().as_secs_f64() / iters as f64
        };

        let gflops = |dim: usize, secs: f64| -> f64 {
            (2.0 * dim as f64 * dim as f64 * dim as f64) / secs / 1e9
        };

        let naive_grid = (n.div_ceil(32), m.div_ceil(32), 1);
        let naive_group = (32, 32, 1);
        let blocked_grid = (n.div_ceil(32), m.div_ceil(32), 1);
        let blocked_group = (32, 8, 1);

        let t_naive = bench_mm(&pipe_naive, &a, &b, &c, naive_grid, naive_group, iters);
        let t_tiled = bench_mm(&pipe_tiled, &a, &b, &c, naive_grid, naive_group, iters);
        let t_blocked = bench_mm(
            &pipe_blocked,
            &a,
            &b,
            &c,
            blocked_grid,
            blocked_group,
            iters,
        );

        // fp16 buffers (half the size)
        let a16 = device.new_buffer(m * k * 2)?;
        let b16 = device.new_buffer(k * n * 2)?;
        let c16 = device.new_buffer(m * n * 2)?;
        a16.with_data_mut(|d| {
            let dst = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, m * k) };
            for v in dst.iter_mut() {
                *v = metal::f32_to_fp16(0.01);
            }
        });
        b16.with_data_mut(|d| {
            let dst = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, k * n) };
            for v in dst.iter_mut() {
                *v = metal::f32_to_fp16(0.01);
            }
        });
        let fp16_grid = (n.div_ceil(64), m.div_ceil(64), 1);
        let fp16_group = (256, 1, 1);
        let t_fp16 = bench_mm(&pipe_fp16, &a16, &b16, &c16, fp16_grid, fp16_group, iters);

        println!(
            "Matmul {}x{}: naive {:.2}ms ({:.0}) | tiled {:.2}ms ({:.0}) | blocked {:.2}ms ({:.0}) | fp16 {:.2}ms ({:.0} GFLOPS)",
            dim, dim,
            t_naive * 1000.0, gflops(dim, t_naive),
            t_tiled * 1000.0, gflops(dim, t_tiled),
            t_blocked * 1000.0, gflops(dim, t_blocked),
            t_fp16 * 1000.0, gflops(dim, t_fp16),
        );
    }

    // ── Bit-shift optimized fp16 matmul (best kernel) ──
    println!("\n=== Bit-shift optimized fp16 matmul (2048x2048) ===");
    {
        let dim = 2048usize;
        let (m, n, k) = (dim, dim, dim);
        let a = device.new_buffer(m * k * 2)?;
        let b = device.new_buffer(k * n * 2)?;
        let c = device.new_buffer(m * n * 2)?;
        a.with_data_mut(|d| {
            let s = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, m * k) };
            for v in s.iter_mut() {
                *v = metal::f32_to_fp16(0.01);
            }
        });
        b.with_data_mut(|d| {
            let s = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, k * n) };
            for v in s.iter_mut() {
                *v = metal::f32_to_fp16(0.01);
            }
        });

        #[repr(C)]
        struct P2 {
            m: u32,
            n: u32,
            k: u32,
        }
        let params = P2 {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let pb = unsafe {
            std::slice::from_raw_parts(&params as *const P2 as *const u8, std::mem::size_of::<P2>())
        };

        let opt_src = r#"
            #include <metal_stdlib>
            #include <metal_simdgroup_matrix>
            using namespace metal;
            struct P { uint M; uint N; uint K; };

            kernel void gemm_opt(device const half *A [[buffer(0)]],
                                 device const half *B [[buffer(1)]],
                                 device half *C       [[buffer(2)]],
                                 constant P &p [[buffer(3)]],
                                 uint2 group_id [[threadgroup_position_in_grid]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint lid [[thread_index_in_threadgroup]]) {

                // BM=64 BN=64 BK=32, padding=4, 16 sg, 512 threads
                threadgroup half tA[64][36];  // 32+4
                threadgroup half tB[32][68];  // 64+4
                uint grow = group_id.y << 6u;  // *64
                uint gcol = group_id.x << 6u;
                uint sg_row = (sgid >> 2u) << 4u;  // (sgid/4)*16
                uint sg_col = (sgid & 3u) << 4u;   // (sgid%4)*16

                simdgroup_half8x8 acc[2][2];
                for (uint i = 0; i < 2u; i++)
                    for (uint j = 0; j < 2u; j++)
                        acc[i][j] = make_filled_simdgroup_matrix<half, 8>(half(0));

                device const half4 *A4 = (device const half4 *)A;
                device const half4 *B4 = (device const half4 *)B;
                uint a4s = p.K >> 2u;  // K/4
                uint b4s = p.N >> 2u;  // N/4

                for (uint t = 0; t < p.K; t += 32u) {
                    {
                        uint r = lid >> 3u;
                        uint c4 = lid & 7u;
                        half4 v = A4[(grow + r) * a4s + (t >> 2u) + c4];
                        uint bc = c4 << 2u;
                        tA[r][bc] = v.x; tA[r][bc+1] = v.y; tA[r][bc+2] = v.z; tA[r][bc+3] = v.w;
                    }
                    {
                        uint r = lid >> 4u;
                        uint c4 = lid & 15u;
                        half4 v = B4[(t + r) * b4s + (gcol >> 2u) + c4];
                        uint bc = c4 << 2u;
                        tB[r][bc] = v.x; tB[r][bc+1] = v.y; tB[r][bc+2] = v.z; tB[r][bc+3] = v.w;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_half8x8 at0, at1, bt0, bt1;
                    simdgroup_load(at0, &tA[sg_row][0], 36);
                    simdgroup_load(at1, &tA[sg_row + 8u][0], 36);
                    simdgroup_load(bt0, &tB[0][sg_col], 68);
                    simdgroup_load(bt1, &tB[0][sg_col + 8u], 68);
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                    simdgroup_load(at0, &tA[sg_row][8], 36);
                    simdgroup_load(at1, &tA[sg_row + 8u][8], 36);
                    simdgroup_load(bt0, &tB[8][sg_col], 68);
                    simdgroup_load(bt1, &tB[8][sg_col + 8u], 68);
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                    simdgroup_load(at0, &tA[sg_row][16], 36);
                    simdgroup_load(at1, &tA[sg_row + 8u][16], 36);
                    simdgroup_load(bt0, &tB[16][sg_col], 68);
                    simdgroup_load(bt1, &tB[16][sg_col + 8u], 68);
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                    simdgroup_load(at0, &tA[sg_row][24], 36);
                    simdgroup_load(at1, &tA[sg_row + 8u][24], 36);
                    simdgroup_load(bt0, &tB[24][sg_col], 68);
                    simdgroup_load(bt1, &tB[24][sg_col + 8u], 68);
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
        let grid = (dim.div_ceil(64), dim.div_ceil(64), 1);
        let group = (512, 1, 1);
        match compile_and_bench(
            &device, &queue, opt_src, "gemm_opt", &a, &b, &c, pb, dim, grid, group,
        ) {
            Ok(gf) => println!(
                "  64x64 BK=32 bitshift+unrolled (16 sg, 512 thr): {:.0} GFLOPS",
                gf
            ),
            Err(e) => println!("  opt: FAIL: {e}"),
        }

        // ── Persistent kernel: 16 threadgroups loop over tiles ──
        println!("\n  === Persistent kernel (16 TG, loop over tiles) ===");
        {
            let persistent_src = r#"
                #include <metal_stdlib>
                #include <metal_simdgroup_matrix>
                using namespace metal;
                struct P { uint M; uint N; uint K; uint tiles_n; uint total_tiles; };

                kernel void gemm_persistent(
                    device const half *A [[buffer(0)]],
                    device const half *B [[buffer(1)]],
                    device half *C       [[buffer(2)]],
                    constant P &p        [[buffer(3)]],
                    uint tg_id [[threadgroup_position_in_grid]],
                    uint sgid [[simdgroup_index_in_threadgroup]],
                    uint lid [[thread_index_in_threadgroup]]) {

                    threadgroup half tA[64][36];
                    threadgroup half tB[32][68];

                    device const half4 *A4 = (device const half4 *)A;
                    device const half4 *B4 = (device const half4 *)B;
                    uint a4s = p.K >> 2u;
                    uint b4s = p.N >> 2u;

                    uint sg_row = (sgid >> 2u) << 4u;
                    uint sg_col = (sgid & 3u) << 4u;

                    // Loop over tiles assigned to this threadgroup
                    for (uint tile = tg_id; tile < p.total_tiles; tile += 16u) {
                        uint tile_y = tile / p.tiles_n;
                        uint tile_x = tile - tile_y * p.tiles_n;
                        uint grow = tile_y << 6u;
                        uint gcol = tile_x << 6u;

                        simdgroup_half8x8 acc[2][2];
                        for (uint i = 0; i < 2u; i++)
                            for (uint j = 0; j < 2u; j++)
                                acc[i][j] = make_filled_simdgroup_matrix<half, 8>(half(0));

                        for (uint t = 0; t < p.K; t += 32u) {
                            {
                                uint r = lid >> 3u;
                                uint c4 = lid & 7u;
                                half4 v = A4[(grow + r) * a4s + (t >> 2u) + c4];
                                uint bc = c4 << 2u;
                                tA[r][bc] = v.x; tA[r][bc+1] = v.y;
                                tA[r][bc+2] = v.z; tA[r][bc+3] = v.w;
                            }
                            {
                                uint r = lid >> 4u;
                                uint c4 = lid & 15u;
                                half4 v = B4[(t + r) * b4s + (gcol >> 2u) + c4];
                                uint bc = c4 << 2u;
                                tB[r][bc] = v.x; tB[r][bc+1] = v.y;
                                tB[r][bc+2] = v.z; tB[r][bc+3] = v.w;
                            }
                            threadgroup_barrier(mem_flags::mem_threadgroup);

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

                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
            "#;

            #[repr(C)]
            struct Pp {
                m: u32,
                n: u32,
                k: u32,
                tiles_n: u32,
                total_tiles: u32,
            }
            let tiles_n = dim >> 6;
            let tiles_m = dim >> 6;
            let total_tiles = tiles_m * tiles_n;
            let pp = Pp {
                m: dim as u32,
                n: dim as u32,
                k: dim as u32,
                tiles_n: tiles_n as u32,
                total_tiles: total_tiles as u32,
            };
            let ppb = unsafe {
                std::slice::from_raw_parts(&pp as *const Pp as *const u8, std::mem::size_of::<Pp>())
            };

            let lib = device.new_library_with_source(persistent_src)?;
            let pipe = device.new_compute_pipeline(&lib.get_function("gemm_persistent")?)?;

            // 16 threadgroups (1 per GPU core), each loops over ~64 tiles
            let grid = (16, 1, 1);
            let group = (512, 1, 1);

            let gf = compile_and_bench(
                &device,
                &queue,
                persistent_src,
                "gemm_persistent",
                &a,
                &b,
                &c,
                ppb,
                dim,
                grid,
                group,
            )?;
            println!("  persistent (16 TG, loop): {:.0} GFLOPS", gf);

            // Also try 32 and 48 threadgroups
            for &ntg in &[32usize, 48, 64] {
                let grid2 = (ntg, 1, 1);
                let gf2 = compile_and_bench(
                    &device,
                    &queue,
                    persistent_src,
                    "gemm_persistent",
                    &a,
                    &b,
                    &c,
                    ppb,
                    dim,
                    grid2,
                    group,
                )?;
                println!("  persistent ({ntg} TG, loop): {:.0} GFLOPS", gf2);
            }
        }

        // ── Last mile: occupancy tuning ──
        println!("\n  === LAST MILE: occupancy + pipelining ===");
        bench_last_mile(&device, &queue, pb, dim, &a, &b, &c)?;
    }

    Ok(())
}

/// Last mile: hand-tuned kernels targeting maximum occupancy.
fn bench_last_mile(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    pb: &[u8],
    dim: usize,
    a: &metal::MtlBuffer,
    b: &metal::MtlBuffer,
    c: &metal::MtlBuffer,
) -> Result<(), MetalError> {
    let mut best = 0.0f64;
    let grid = (dim >> 6, dim >> 6, 1);

    // BK=16, 16 sg (512 thr) — 2x more occupancy than BK=32
    let kernel_bk16 = make_kernel(64, 64, 16, 4, 16, 512);
    match compile_and_bench(
        device,
        queue,
        &kernel_bk16,
        "gemm_lm",
        a,
        b,
        c,
        pb,
        dim,
        grid,
        (512, 1, 1),
    ) {
        Ok(gf) => {
            let m = if gf > best {
                best = gf;
                "  BEST"
            } else {
                ""
            };
            println!("    BK=16 16sg 512thr (TG=4.7KB): {:.0} GFLOPS{m}", gf);
        }
        Err(e) => println!("    BK=16 16sg: {e}"),
    }

    // BK=32, 16 sg (512 thr) — baseline
    let kernel_bk32 = make_kernel(64, 64, 32, 4, 16, 512);
    match compile_and_bench(
        device,
        queue,
        &kernel_bk32,
        "gemm_lm",
        a,
        b,
        c,
        pb,
        dim,
        grid,
        (512, 1, 1),
    ) {
        Ok(gf) => {
            let m = if gf > best {
                best = gf;
                "  BEST"
            } else {
                ""
            };
            println!("    BK=32 16sg 512thr (TG=9.0KB): {:.0} GFLOPS{m}", gf);
        }
        Err(e) => println!("    BK=32 16sg: {e}"),
    }

    // BK=16, smaller tile 32x32, 4 sg (128 thr) — max occupancy per core
    let kernel_32x32 = make_kernel(32, 32, 16, 4, 4, 128);
    let grid32 = (dim >> 5, dim >> 5, 1);
    match compile_and_bench(
        device,
        queue,
        &kernel_32x32,
        "gemm_lm",
        a,
        b,
        c,
        pb,
        dim,
        grid32,
        (128, 1, 1),
    ) {
        Ok(gf) => {
            let m = if gf > best {
                best = gf;
                "  BEST"
            } else {
                ""
            };
            println!("    32x32 BK=16 4sg 128thr (TG=1.4KB): {:.0} GFLOPS{m}", gf);
        }
        Err(e) => println!("    32x32: {e}"),
    }

    // BK=32, smaller tile 32x32, 4 sg (128 thr)
    let kernel_32x32_bk32 = make_kernel(32, 32, 32, 4, 4, 128);
    match compile_and_bench(
        device,
        queue,
        &kernel_32x32_bk32,
        "gemm_lm",
        a,
        b,
        c,
        pb,
        dim,
        grid32,
        (128, 1, 1),
    ) {
        Ok(gf) => {
            let m = if gf > best {
                best = gf;
                "  BEST"
            } else {
                ""
            };
            println!("    32x32 BK=32 4sg 128thr (TG=2.6KB): {:.0} GFLOPS{m}", gf);
        }
        Err(e) => println!("    32x32 bk32: {e}"),
    }

    // BK=8, 16 sg (512 thr) — maximum occupancy
    let kernel_bk8 = make_kernel(64, 64, 8, 4, 16, 512);
    match compile_and_bench(
        device,
        queue,
        &kernel_bk8,
        "gemm_lm",
        a,
        b,
        c,
        pb,
        dim,
        grid,
        (512, 1, 1),
    ) {
        Ok(gf) => {
            let m = if gf > best {
                best = gf;
                "  BEST"
            } else {
                ""
            };
            println!("    BK=8  16sg 512thr (TG=2.6KB): {:.0} GFLOPS{m}", gf);
        }
        Err(e) => println!("    BK=8 16sg: {e}"),
    }

    // Pipelined: submit 4 command buffers before waiting
    println!("    --- pipelined (4 cmd bufs) ---");
    for &bk in &[16usize, 32] {
        let kernel = make_kernel(64, 64, bk, 4, 16, 512);
        let lib = device.new_library_with_source(&kernel)?;
        let pipe = device.new_compute_pipeline(&lib.get_function("gemm_lm")?)?;
        let iters = 10;

        for _ in 0..3 {
            let cmd = queue.command_buffer()?;
            let enc = cmd.compute_encoder()?;
            enc.set_pipeline(&pipe);
            enc.set_buffer(a, 0, 0);
            enc.set_buffer(b, 0, 1);
            enc.set_buffer(c, 0, 2);
            enc.set_bytes(pb, 3);
            enc.dispatch_threadgroups(grid, (512, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let t0 = Instant::now();
        for batch in 0..iters {
            let cmd = queue.command_buffer()?;
            let enc = cmd.compute_encoder()?;
            enc.set_pipeline(&pipe);
            enc.set_buffer(a, 0, 0);
            enc.set_buffer(b, 0, 1);
            enc.set_buffer(c, 0, 2);
            enc.set_bytes(pb, 3);
            enc.dispatch_threadgroups(grid, (512, 1, 1));
            enc.end_encoding();
            cmd.commit();
            if batch % 4 == 3 || batch == iters - 1 {
                cmd.wait_until_completed();
            }
        }
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        let gf = (2.0 * dim as f64 * dim as f64 * dim as f64) / dt / 1e9;
        let m = if gf > best {
            best = gf;
            "  BEST"
        } else {
            ""
        };
        println!("    BK={bk:<2} pipelined 16sg: {:.0} GFLOPS{m}", gf);
    }

    println!("\n  OVERALL BEST: {:.0} GFLOPS", best);
    Ok(())
}

/// Generate an optimized kernel with bit-shift addressing and manual unroll.
fn make_kernel(
    bm: usize,
    bn: usize,
    bk: usize,
    pad: usize,
    _num_sg: usize,
    num_thr: usize,
) -> String {
    let sg_tm = 2usize;
    let sg_tn = 2usize;
    let sg_cols = bn / (sg_tn * 8);
    let bn_shift = bn.trailing_zeros();
    let bk_v4 = bk >> 2;
    let bn_v4 = bn >> 2;

    let mut mma_unrolled = String::new();
    for kk in (0..bk).step_by(8) {
        mma_unrolled.push_str(&format!(
            "simdgroup_load(at0, &tA[sg_row][{kk}], {stride_a});\n\
             simdgroup_load(at1, &tA[sg_row+8u][{kk}], {stride_a});\n\
             simdgroup_load(bt0, &tB[{kk}][sg_col], {stride_b});\n\
             simdgroup_load(bt1, &tB[{kk}][sg_col+8u], {stride_b});\n\
             simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);\n\
             simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);\n\
             simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);\n\
             simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);\n",
            kk = kk,
            stride_a = bk + pad,
            stride_b = bn + pad,
        ));
    }

    format!(
        r#"
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        struct P {{ uint M; uint N; uint K; }};

        kernel void gemm_lm(device const half *A [[buffer(0)]],
                             device const half *B [[buffer(1)]],
                             device half *C       [[buffer(2)]],
                             constant P &p [[buffer(3)]],
                             uint2 group_id [[threadgroup_position_in_grid]],
                             uint sgid [[simdgroup_index_in_threadgroup]],
                             uint lid [[thread_index_in_threadgroup]]) {{

            threadgroup half tA[{bm}][{bk_pad}];
            threadgroup half tB[{bk}][{bn_pad}];
            uint grow = group_id.y << {bm_shift}u;
            uint gcol = group_id.x << {bn_shift}u;
            uint sg_row = (sgid / {sg_cols}u) * {sg_tm_8}u;
            uint sg_col = (sgid % {sg_cols}u) * {sg_tn_8}u;

            simdgroup_half8x8 acc[{sg_tm}][{sg_tn}];
            for (uint i = 0; i < {sg_tm}u; i++)
                for (uint j = 0; j < {sg_tn}u; j++)
                    acc[i][j] = make_filled_simdgroup_matrix<half, 8>(half(0));

            device const half4 *A4 = (device const half4 *)A;
            device const half4 *B4 = (device const half4 *)B;
            uint a4s = p.K >> 2u;
            uint b4s = p.N >> 2u;

            for (uint t = 0; t < p.K; t += {bk}u) {{
                for (uint i = lid; i < {a_total}u; i += {num_thr}u) {{
                    uint r = i / {bk_v4}u, cv = i - r * {bk_v4}u;
                    half4 v = A4[(grow + r) * a4s + (t >> 2u) + cv];
                    uint bc = cv << 2u;
                    tA[r][bc]=v.x; tA[r][bc+1]=v.y; tA[r][bc+2]=v.z; tA[r][bc+3]=v.w;
                }}
                for (uint i = lid; i < {b_total}u; i += {num_thr}u) {{
                    uint r = i / {bn_v4}u, cv = i - r * {bn_v4}u;
                    half4 v = B4[(t + r) * b4s + (gcol >> 2u) + cv];
                    uint bc = cv << 2u;
                    tB[r][bc]=v.x; tB[r][bc+1]=v.y; tB[r][bc+2]=v.z; tB[r][bc+3]=v.w;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                simdgroup_half8x8 at0, at1, bt0, bt1;
                {mma}

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}

            for (uint i = 0; i < {sg_tm}u; i++)
                for (uint j = 0; j < {sg_tn}u; j++)
                    simdgroup_store(acc[i][j],
                        C + (grow + sg_row + i*8u) * p.N + (gcol + sg_col + j*8u), p.N);
        }}
    "#,
        bm = bm,
        bk = bk,
        bk_pad = bk + pad,
        bn_pad = bn + pad,
        bm_shift = bm.trailing_zeros(),
        bn_shift = bn_shift,
        sg_cols = sg_cols,
        sg_tm = sg_tm,
        sg_tn = sg_tn,
        sg_tm_8 = sg_tm * 8,
        sg_tn_8 = sg_tn * 8,
        a_total = bm * bk_v4,
        b_total = bk * bn_v4,
        bk_v4 = bk_v4,
        bn_v4 = bn_v4,
        num_thr = num_thr,
        mma = mma_unrolled,
    )
}

/// Compile MSL kernel, benchmark it, return GFLOPS.
fn compile_and_bench(
    device: &MtlDevice,
    queue: &metal::MtlCommandQueue,
    src: &str,
    func_name: &str,
    a: &metal::MtlBuffer,
    b: &metal::MtlBuffer,
    c: &metal::MtlBuffer,
    params: &[u8],
    dim: usize,
    grid: (usize, usize, usize),
    group: (usize, usize, usize),
) -> Result<f64, MetalError> {
    let lib = device.new_library_with_source(src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function(func_name)?)?;
    let iters = 10;

    for _ in 0..3 {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipe);
        enc.set_buffer(a, 0, 0);
        enc.set_buffer(b, 0, 1);
        enc.set_buffer(c, 0, 2);
        enc.set_bytes(params, 3);
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.command_buffer()?;
        let enc = cmd.compute_encoder()?;
        enc.set_pipeline(&pipe);
        enc.set_buffer(a, 0, 0);
        enc.set_buffer(b, 0, 1);
        enc.set_buffer(c, 0, 2);
        enc.set_bytes(params, 3);
        enc.dispatch_threadgroups(grid, group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    Ok((2.0 * dim as f64 * dim as f64 * dim as f64) / dt / 1e9)
}
