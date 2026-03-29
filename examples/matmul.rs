//! Matrix multiplication on Metal GPU — compute example

use metal::{MetalError, MtlDevice};

fn main() -> Result<(), MetalError> {
    let device = MtlDevice::system_default()?;
    println!("Device: {}", device.name());

    let queue = device.new_command_queue()?;

    // Naive matmul kernel: C[M,N] = A[M,K] * B[K,N]
    let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct MatmulParams {
            uint M;
            uint N;
            uint K;
        };

        kernel void matmul(device const float *A [[buffer(0)]],
                           device const float *B [[buffer(1)]],
                           device float *C       [[buffer(2)]],
                           constant MatmulParams &params [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
            uint row = gid.y;
            uint col = gid.x;
            if (row >= params.M || col >= params.N) return;

            float sum = 0.0;
            for (uint k = 0; k < params.K; k++) {
                sum += A[row * params.K + k] * B[k * params.N + col];
            }
            C[row * params.N + col] = sum;
        }
    "#;

    let lib = device.new_library_with_source(source)?;
    let func = lib.get_function("matmul")?;
    let pipeline = device.new_compute_pipeline(&func)?;

    let m = 64usize;
    let n = 64usize;
    let k = 64usize;

    let buf_a = device.new_buffer(m * k * 4)?;
    let buf_b = device.new_buffer(k * n * 4)?;
    let buf_c = device.new_buffer(m * n * 4)?;

    // A = identity, B = all ones => C should be all ones
    buf_a.with_f32_mut(|d| {
        for i in 0..m {
            for j in 0..k {
                d[i * k + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    });
    buf_b.with_f32_mut(|d| {
        for i in 0..d.len() {
            d[i] = 1.0;
        }
    });

    #[repr(C)]
    struct MatmulParams {
        m: u32,
        n: u32,
        k: u32,
    }
    let params = MatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
    };
    let params_bytes = unsafe {
        std::slice::from_raw_parts(
            &params as *const MatmulParams as *const u8,
            std::mem::size_of::<MatmulParams>(),
        )
    };

    let cmd = queue.command_buffer()?;
    let enc = cmd.compute_encoder()?;
    enc.set_pipeline(&pipeline);
    enc.set_buffer(&buf_a, 0, 0);
    enc.set_buffer(&buf_b, 0, 1);
    enc.set_buffer(&buf_c, 0, 2);
    enc.set_bytes(params_bytes, 3);
    enc.dispatch_threads((n, m, 1), (16, 16, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    buf_c.with_f32(|d| {
        let mut max_err: f32 = 0.0;
        for i in 0..m * n {
            max_err = max_err.max((d[i] - 1.0).abs());
        }
        if max_err < 1e-5 {
            println!(
                "PASS: {}x{}x{} matmul verified (max_err={:.2e})",
                m, n, k, max_err
            );
        } else {
            println!("FAIL: max_err = {:.6}", max_err);
        }
    });

    Ok(())
}
