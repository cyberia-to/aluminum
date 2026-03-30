//! GPU profiler — precise timing, pipeline properties, occupancy analysis
//!
//! Uses MTLCommandBuffer.GPUStartTime/GPUEndTime for accurate GPU-side timing.
//! Reports pipeline properties, threadgroup memory, and estimated occupancy.

use metal::{MetalError, MtlDevice};
use std::time::Instant;

fn main() -> Result<(), MetalError> {
    let device = MtlDevice::system_default()?;
    let queue = device.new_command_queue()?;
    println!("Device: {}", device.name());
    println!("Unified memory: {}", device.has_unified_memory());
    println!(
        "Max buffer: {} MB",
        device.max_buffer_length() / (1024 * 1024)
    );
    println!();

    let kernel_src = r#"
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        struct P { uint M; uint N; uint K; };

        kernel void gemm(device const half *A [[buffer(0)]],
                         device const half *B [[buffer(1)]],
                         device half *C       [[buffer(2)]],
                         constant P &p [[buffer(3)]],
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
            device const half4 *B4 = (device const half4 *)B;
            uint a4s = p.K >> 2u;
            uint b4s = p.N >> 2u;

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
        }
    "#;

    let lib = device.new_library_with_source(kernel_src)?;
    let pipe = device.new_compute_pipeline(&lib.get_function("gemm")?)?;

    // ── Pipeline properties ──
    println!("=== Pipeline Properties ===");
    println!(
        "  max threads/threadgroup: {}",
        pipe.max_total_threads_per_threadgroup()
    );
    println!(
        "  thread execution width:  {}",
        pipe.thread_execution_width()
    );
    println!(
        "  static TG memory:        {} bytes",
        pipe.static_threadgroup_memory_length()
    );

    // Calculated
    let tg_mem = pipe.static_threadgroup_memory_length();
    let max_tg_per_core = 32768 / tg_mem.max(1);
    let threads_per_tg = 512;
    let sg_per_tg = threads_per_tg / 32;
    println!("  threads/TG (our config):  {}", threads_per_tg);
    println!("  simdgroups/TG:            {}", sg_per_tg);
    println!("  max TG/core (TG mem):     {}", max_tg_per_core);
    println!();

    // ── GPU timestamps: precise kernel timing ──
    println!("=== GPU Timing (precise) ===");

    for &dim in &[512usize, 1024, 2048, 4096] {
        let a = device.new_buffer(dim * dim * 2)?;
        let b = device.new_buffer(dim * dim * 2)?;
        let c = device.new_buffer(dim * dim * 2)?;
        a.with_data_mut(|d| {
            let s =
                unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, dim * dim) };
            for v in s.iter_mut() {
                *v = metal::f32_to_fp16(0.001);
            }
        });
        b.with_data_mut(|d| {
            let s =
                unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, dim * dim) };
            for v in s.iter_mut() {
                *v = metal::f32_to_fp16(0.001);
            }
        });

        #[repr(C)]
        struct P {
            m: u32,
            n: u32,
            k: u32,
        }
        let params = P {
            m: dim as u32,
            n: dim as u32,
            k: dim as u32,
        };
        let pb = unsafe {
            std::slice::from_raw_parts(&params as *const P as *const u8, std::mem::size_of::<P>())
        };
        let grid = (dim >> 6, dim >> 6, 1);
        let group = (512, 1, 1);

        // Warmup
        for _ in 0..5 {
            let cmd = queue.command_buffer()?;
            let enc = cmd.compute_encoder()?;
            enc.set_pipeline(&pipe);
            enc.set_buffer(&a, 0, 0);
            enc.set_buffer(&b, 0, 1);
            enc.set_buffer(&c, 0, 2);
            enc.set_bytes(pb, 3);
            enc.dispatch_threadgroups(grid, group);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Measure: CPU time vs GPU time
        let iters = 10;
        let mut gpu_total = 0.0f64;
        let cpu_t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.command_buffer()?;
            let enc = cmd.compute_encoder()?;
            enc.set_pipeline(&pipe);
            enc.set_buffer(&a, 0, 0);
            enc.set_buffer(&b, 0, 1);
            enc.set_buffer(&c, 0, 2);
            enc.set_bytes(pb, 3);
            enc.dispatch_threadgroups(grid, group);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            gpu_total += cmd.gpu_time();
        }
        let cpu_total = cpu_t0.elapsed().as_secs_f64();

        let cpu_per = cpu_total / iters as f64;
        let gpu_per = gpu_total / iters as f64;
        let overhead = cpu_per - gpu_per;
        let flops = 2.0 * dim as f64 * dim as f64 * dim as f64;
        let gflops_cpu = flops / cpu_per / 1e9;
        let gflops_gpu = flops / gpu_per / 1e9;

        let tiles = (dim >> 6) * (dim >> 6);
        let k_iters = dim >> 5; // K/32

        println!("  {dim}×{dim} ({tiles} tiles, {k_iters} K-iters):");
        println!(
            "    CPU time: {:.3} ms → {:.0} GFLOPS",
            cpu_per * 1000.0,
            gflops_cpu
        );
        println!(
            "    GPU time: {:.3} ms → {:.0} GFLOPS",
            gpu_per * 1000.0,
            gflops_gpu
        );
        println!(
            "    overhead: {:.3} ms ({:.1}%)",
            overhead * 1000.0,
            overhead / cpu_per * 100.0
        );
        println!("    per-tile GPU: {:.1} μs", gpu_per * 1e6 / tiles as f64);
        println!(
            "    per-K-iter:   {:.1} μs (includes barrier)",
            gpu_per * 1e6 / (tiles * k_iters) as f64
        );
    }

    // ── Occupancy sweep: vary TG memory padding ──
    println!("\n=== Occupancy Sweep (2048×2048) ===");
    println!("  Testing different threadgroup memory layouts:\n");

    let dim = 2048usize;
    let a = device.new_buffer(dim * dim * 2)?;
    let b = device.new_buffer(dim * dim * 2)?;
    let c = device.new_buffer(dim * dim * 2)?;
    a.with_data_mut(|d| {
        let s = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, dim * dim) };
        for v in s.iter_mut() {
            *v = metal::f32_to_fp16(0.001);
        }
    });
    b.with_data_mut(|d| {
        let s = unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr() as *mut u16, dim * dim) };
        for v in s.iter_mut() {
            *v = metal::f32_to_fp16(0.001);
        }
    });
    #[repr(C)]
    struct Ps {
        m: u32,
        n: u32,
        k: u32,
    }
    let params = Ps {
        m: dim as u32,
        n: dim as u32,
        k: dim as u32,
    };
    let pb = unsafe {
        std::slice::from_raw_parts(&params as *const Ps as *const u8, std::mem::size_of::<Ps>())
    };

    // (pad_a, pad_b, label)
    let configs: &[(usize, usize, &str)] = &[
        (4, 4, "pad+4/+4 (baseline, 8960B)"),
        (2, 4, "pad+2/+4 (8576B)"),
        (2, 2, "pad+2/+2 (8448B)"),
        (1, 2, "pad+1/+2 (8288B)"),
        (2, 0, "pad+2/+0 (8448B)"),
        (0, 2, "pad+0/+2 (8320B)"),
        (1, 0, "pad+1/+0 (8256B)"),
        (0, 0, "pad+0/+0 (8192B)"),
    ];

    for &(pad_a, pad_b, label) in configs {
        let stride_a = 32 + pad_a;
        let stride_b = 64 + pad_b;
        let tg_bytes = 64 * stride_a * 2 + 32 * stride_b * 2;
        let max_tg = 32768 / tg_bytes;

        let src = format!(
            r#"
            #include <metal_stdlib>
            #include <metal_simdgroup_matrix>
            using namespace metal;
            struct P {{ uint M; uint N; uint K; }};

            kernel void gemm_occ(device const half *A [[buffer(0)]],
                                 device const half *B [[buffer(1)]],
                                 device half *C       [[buffer(2)]],
                                 constant P &p [[buffer(3)]],
                                 uint2 group_id [[threadgroup_position_in_grid]],
                                 uint sgid [[simdgroup_index_in_threadgroup]],
                                 uint lid [[thread_index_in_threadgroup]]) {{

                threadgroup half tA[64][{sa}];
                threadgroup half tB[32][{sb}];
                uint grow = group_id.y << 6u;
                uint gcol = group_id.x << 6u;
                uint sg_row = (sgid >> 2u) << 4u;
                uint sg_col = (sgid & 3u) << 4u;

                simdgroup_half8x8 acc[2][2];
                for (uint i = 0; i < 2u; i++)
                    for (uint j = 0; j < 2u; j++)
                        acc[i][j] = make_filled_simdgroup_matrix<half, 8>(half(0));

                device const half4 *A4 = (device const half4 *)A;
                device const half4 *B4 = (device const half4 *)B;
                uint a4s = p.K >> 2u;
                uint b4s = p.N >> 2u;

                for (uint t = 0; t < p.K; t += 32u) {{
                    {{
                        uint r = lid >> 3u;
                        uint c4 = lid & 7u;
                        half4 v = A4[(grow + r) * a4s + (t >> 2u) + c4];
                        uint bc = c4 << 2u;
                        tA[r][bc] = v.x; tA[r][bc+1] = v.y;
                        tA[r][bc+2] = v.z; tA[r][bc+3] = v.w;
                    }}
                    {{
                        uint r = lid >> 4u;
                        uint c4 = lid & 15u;
                        half4 v = B4[(t + r) * b4s + (gcol >> 2u) + c4];
                        uint bc = c4 << 2u;
                        tB[r][bc] = v.x; tB[r][bc+1] = v.y;
                        tB[r][bc+2] = v.z; tB[r][bc+3] = v.w;
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    simdgroup_half8x8 at0, at1, bt0, bt1;
                    simdgroup_load(at0, &tA[sg_row][0], {sa});
                    simdgroup_load(at1, &tA[sg_row+8u][0], {sa});
                    simdgroup_load(bt0, &tB[0][sg_col], {sb});
                    simdgroup_load(bt1, &tB[0][sg_col+8u], {sb});
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                    simdgroup_load(at0, &tA[sg_row][8], {sa});
                    simdgroup_load(at1, &tA[sg_row+8u][8], {sa});
                    simdgroup_load(bt0, &tB[8][sg_col], {sb});
                    simdgroup_load(bt1, &tB[8][sg_col+8u], {sb});
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                    simdgroup_load(at0, &tA[sg_row][16], {sa});
                    simdgroup_load(at1, &tA[sg_row+8u][16], {sa});
                    simdgroup_load(bt0, &tB[16][sg_col], {sb});
                    simdgroup_load(bt1, &tB[16][sg_col+8u], {sb});
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                    simdgroup_load(at0, &tA[sg_row][24], {sa});
                    simdgroup_load(at1, &tA[sg_row+8u][24], {sa});
                    simdgroup_load(bt0, &tB[24][sg_col], {sb});
                    simdgroup_load(bt1, &tB[24][sg_col+8u], {sb});
                    simdgroup_multiply_accumulate(acc[0][0], at0, bt0, acc[0][0]);
                    simdgroup_multiply_accumulate(acc[0][1], at0, bt1, acc[0][1]);
                    simdgroup_multiply_accumulate(acc[1][0], at1, bt0, acc[1][0]);
                    simdgroup_multiply_accumulate(acc[1][1], at1, bt1, acc[1][1]);

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}

                for (uint i = 0; i < 2u; i++)
                    for (uint j = 0; j < 2u; j++)
                        simdgroup_store(acc[i][j],
                            C + (grow + sg_row + i*8u) * p.N + (gcol + sg_col + j*8u), p.N);
            }}
        "#,
            sa = stride_a,
            sb = stride_b
        );

        let lib = device.new_library_with_source(&src)?;
        let pipe = device.new_compute_pipeline(&lib.get_function("gemm_occ")?)?;
        let actual_tg = pipe.static_threadgroup_memory_length();

        let grid = (dim >> 6, dim >> 6, 1);
        let group = (512, 1, 1);

        // Warmup
        for _ in 0..5 {
            let cmd = queue.command_buffer()?;
            let enc = cmd.compute_encoder()?;
            enc.set_pipeline(&pipe);
            enc.set_buffer(&a, 0, 0);
            enc.set_buffer(&b, 0, 1);
            enc.set_buffer(&c, 0, 2);
            enc.set_bytes(pb, 3);
            enc.dispatch_threadgroups(grid, group);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let iters = 10;
        let mut gpu_total = 0.0f64;
        for _ in 0..iters {
            let cmd = queue.command_buffer()?;
            let enc = cmd.compute_encoder()?;
            enc.set_pipeline(&pipe);
            enc.set_buffer(&a, 0, 0);
            enc.set_buffer(&b, 0, 1);
            enc.set_buffer(&c, 0, 2);
            enc.set_bytes(pb, 3);
            enc.dispatch_threadgroups(grid, group);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            gpu_total += cmd.gpu_time();
        }
        let gpu_per = gpu_total / iters as f64;
        let gf = (2.0 * dim as f64 * dim as f64 * dim as f64) / gpu_per / 1e9;

        let marker = if gf > 3650.0 { "← !" } else { "" };
        println!("  {label}");
        println!(
            "    actual={actual_tg}B, max_tg/core={max_tg}, GPU: {:.0} GFLOPS {marker}\n",
            gf
        );
    }

    Ok(())
}
