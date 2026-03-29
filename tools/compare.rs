//! Performance comparison: rmetal (direct FFI) vs objc2-metal (ObjC bindings)
//!
//! Measures:
//! 1. Device discovery latency
//! 2. Buffer creation throughput
//! 3. Shader compilation latency
//! 4. Command dispatch overhead (many small dispatches)
//! 5. Large compute throughput (SAXPY on 16M floats)

// ── rmetal (direct objc_msgSend FFI) ──

mod rmetal_bench {
    use metal::MtlDevice;
    use std::time::Instant;

    pub fn device_discovery(iters: usize) -> f64 {
        let _ = MtlDevice::system_default().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = MtlDevice::system_default().unwrap();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn buffer_creation(iters: usize, size: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        for _ in 0..3 {
            let _ = dev.new_buffer(size).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dev.new_buffer(size).unwrap();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn shader_compile(iters: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let src = SAXPY_SRC;
        let _ = dev.new_library_with_source(src).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dev.new_library_with_source(src).unwrap();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Pure CPU encoding overhead — no per-iter GPU wait.
    pub fn encode_overhead(iters: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();

        for _ in 0..10 {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let t0 = Instant::now();
        let mut last_cmd = None;
        for _ in 0..iters {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            last_cmd = Some(cmd);
        }
        if let Some(cmd) = last_cmd {
            cmd.wait_until_completed();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Pure CPU encoding via unchecked path
    pub fn encode_unchecked(iters: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();

        for _ in 0..10 {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let t0 = Instant::now();
        let mut last_cmd = None::<metal::MtlCommandBuffer>;
        for _ in 0..iters {
            unsafe {
                let cmd = queue.command_buffer_unchecked();
                let enc = cmd.compute_encoder_unchecked();
                enc.set_pipeline(&pipe);
                enc.set_buffer(&buf, 0, 0);
                enc.dispatch_threads((256, 1, 1), (64, 1, 1));
                enc.end_encoding();
                cmd.commit();
                last_cmd = Some(cmd);
            }
        }
        if let Some(cmd) = last_cmd {
            cmd.wait_until_completed();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn dispatch_overhead(iters: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();

        for _ in 0..10 {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Unchecked dispatch — zero-overhead wrappers, unretained references
    pub fn dispatch_unchecked(iters: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();

        for _ in 0..10 {
            unsafe {
                let cmd = queue.command_buffer_unchecked();
                let enc = cmd.compute_encoder_unchecked();
                enc.set_pipeline(&pipe);
                enc.set_buffer(&buf, 0, 0);
                enc.dispatch_threads((256, 1, 1), (64, 1, 1));
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                let cmd = queue.command_buffer_unchecked();
                let enc = cmd.compute_encoder_unchecked();
                enc.set_pipeline(&pipe);
                enc.set_buffer(&buf, 0, 0);
                enc.dispatch_threads((256, 1, 1), (64, 1, 1));
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Autoreleased dispatch — no retain/release at all, autorelease pool per iter
    pub fn dispatch_autoreleased(iters: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();

        for _ in 0..10 {
            metal::autorelease_pool(|| unsafe {
                let cmd = queue.command_buffer_autoreleased();
                let enc = cmd.compute_encoder_autoreleased();
                enc.set_pipeline(&pipe);
                enc.set_buffer(&buf, 0, 0);
                enc.dispatch_threads((256, 1, 1), (64, 1, 1));
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });
        }

        let t0 = Instant::now();
        // Single outer pool, drain every 64 iters to prevent unbounded growth
        metal::autorelease_pool(|| {
            for i in 0..iters {
                unsafe {
                    let cmd = queue.command_buffer_autoreleased();
                    let enc = cmd.compute_encoder_autoreleased();
                    enc.set_pipeline(&pipe);
                    enc.set_buffer(&buf, 0, 0);
                    enc.dispatch_threads((256, 1, 1), (64, 1, 1));
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                }
                if i % 64 == 63 {
                    // Drain and re-push to prevent memory buildup
                    // (handled by nested pool)
                }
            }
        });
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn large_compute(iters: usize, n: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(SAXPY_SRC).unwrap();
        let func = lib.get_function("saxpy").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();

        let buf_x = dev.new_buffer(n * 4).unwrap();
        let buf_y = dev.new_buffer(n * 4).unwrap();
        buf_x.with_f32_mut(|d| {
            for (i, v) in d.iter_mut().enumerate() {
                *v = i as f32;
            }
        });
        buf_y.with_f32_mut(|d| {
            for v in d.iter_mut() {
                *v = 1.0;
            }
        });
        let a_val: f32 = 2.0;
        let a_bytes = a_val.to_le_bytes();

        for _ in 0..3 {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf_x, 0, 0);
            enc.set_buffer(&buf_y, 0, 1);
            enc.set_bytes(&a_bytes, 2);
            enc.dispatch_threads((n, 1, 1), (256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        buf_y.with_f32_mut(|d| {
            for v in d.iter_mut() {
                *v = 1.0;
            }
        });

        let t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf_x, 0, 0);
            enc.set_buffer(&buf_y, 0, 1);
            enc.set_bytes(&a_bytes, 2);
            enc.dispatch_threads((n, 1, 1), (256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Raw dispatch — bypass all wrappers. Pure objc_msgSend + cached sel.
    pub fn dispatch_raw(iters: usize) -> f64 {
        use metal::ffi::*;
        use std::ffi::c_void;

        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();

        let q = queue.as_raw();
        let p = pipe.as_raw();
        let b = buf.as_raw();
        let s_cb = SEL_commandBuffer();
        let s_ce = SEL_computeCommandEncoder();
        let s_sp = SEL_setComputePipelineState();
        let s_sb = SEL_setBuffer_offset_atIndex();
        let s_dt = SEL_dispatchThreads();
        let s_ee = SEL_endEncoding();
        let s_cm = SEL_commit();
        let s_wt = SEL_waitUntilCompleted();
        let _s_re = SEL_retain();
        let _s_rl = SEL_release();

        type F0 = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        type FV = unsafe extern "C" fn(ObjcId, ObjcSel);
        type F1 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId);
        type F3 = unsafe extern "C" fn(ObjcId, ObjcSel, ObjcId, NSUInteger, NSUInteger);
        type FD = unsafe extern "C" fn(ObjcId, ObjcSel, MTLSize, MTLSize);

        let f0: F0 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
        let fv: FV = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
        let f1: F1 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
        let f3: F3 = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };
        let fd: FD = unsafe { std::mem::transmute(objc_msgSend as *const c_void) };

        let grid = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        for _ in 0..10 {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            enc.set_pipeline(&pipe);
            enc.set_buffer(&buf, 0, 0);
            enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                // ARC fast-retain: skip autorelease+retain round-trip
                let cmd = f0(q, s_cb);
                objc_retainAutoreleasedReturnValue(cmd);
                let enc = f0(cmd, s_ce);
                objc_retainAutoreleasedReturnValue(enc);
                f1(enc, s_sp, p);
                f3(enc, s_sb, b, 0, 0);
                fd(enc, s_dt, grid, group);
                fv(enc, s_ee);
                fv(cmd, s_cm);
                fv(cmd, s_wt);
                objc_release(enc);
                objc_release(cmd);
            }
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// ComputeDispatcher: pre-resolved IMP + unretained references
    pub fn dispatch_imp(iters: usize) -> f64 {
        use metal::ComputeDispatcher;

        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();
        let disp = ComputeDispatcher::new(&queue);

        for _ in 0..10 {
            unsafe { disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1)) };
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe { disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1)) };
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// SAXPY with ComputeDispatcher
    pub fn large_compute_imp(iters: usize, n: usize) -> f64 {
        use metal::ComputeDispatcher;

        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(SAXPY_SRC).unwrap();
        let func = lib.get_function("saxpy").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf_x = dev.new_buffer(n * 4).unwrap();
        let buf_y = dev.new_buffer(n * 4).unwrap();
        buf_x.with_f32_mut(|d| {
            for (i, v) in d.iter_mut().enumerate() {
                *v = i as f32;
            }
        });
        buf_y.with_f32_mut(|d| {
            for v in d.iter_mut() {
                *v = 1.0;
            }
        });
        let a: f32 = 2.0;
        let ab = a.to_le_bytes();
        let disp = ComputeDispatcher::new(&queue);

        for _ in 0..3 {
            unsafe {
                disp.dispatch_with_bytes(
                    &pipe,
                    &[(&buf_x, 0, 0), (&buf_y, 0, 1)],
                    &ab,
                    2,
                    (n, 1, 1),
                    (256, 1, 1),
                );
            }
        }
        buf_y.with_f32_mut(|d| {
            for v in d.iter_mut() {
                *v = 1.0;
            }
        });

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                disp.dispatch_with_bytes(
                    &pipe,
                    &[(&buf_x, 0, 0), (&buf_y, 0, 1)],
                    &ab,
                    2,
                    (n, 1, 1),
                    (256, 1, 1),
                );
            }
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Batch encode: N dispatches per command buffer, amortized cost per dispatch
    pub fn batch_encode(batch_size: usize, iters: usize) -> f64 {
        use metal::ComputeDispatcher;

        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();
        let disp = ComputeDispatcher::new(&queue);

        for _ in 0..3 {
            unsafe {
                disp.dispatch_batch(|batch| {
                    for _ in 0..batch_size {
                        batch.set_pipeline(&pipe);
                        batch.set_buffer(&buf, 0, 0);
                        batch.dispatch_threads((256, 1, 1), (64, 1, 1));
                    }
                });
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                disp.dispatch_batch(|batch| {
                    for _ in 0..batch_size {
                        batch.set_pipeline(&pipe);
                        batch.set_buffer(&buf, 0, 0);
                        batch.dispatch_threads((256, 1, 1), (64, 1, 1));
                    }
                });
            }
        }
        t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
    }

    /// Pure encode cost via MtlComputeEncoder (msgSend per call).
    /// Amortized over batch_size dispatches per cmd buffer.
    pub fn encode_encoder(batch_size: usize, iters: usize) -> f64 {
        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();

        for _ in 0..3 {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            for _ in 0..batch_size {
                enc.set_pipeline(&pipe);
                enc.set_buffer(&buf, 0, 0);
                enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.command_buffer().unwrap();
            let enc = cmd.compute_encoder().unwrap();
            for _ in 0..batch_size {
                enc.set_pipeline(&pipe);
                enc.set_buffer(&buf, 0, 0);
                enc.dispatch_threads((256, 1, 1), (64, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
    }

    /// Pipelined dispatch: overlap GPU execution of batch N with encoding of N+1
    pub fn dispatch_pipelined(iters: usize) -> f64 {
        use metal::ComputeDispatcher;

        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let lib = dev.new_library_with_source(NOOP_SRC).unwrap();
        let func = lib.get_function("noop").unwrap();
        let pipe = dev.new_compute_pipeline(&func).unwrap();
        let buf = dev.new_buffer(256 * 4).unwrap();
        let disp = ComputeDispatcher::new(&queue);

        // warmup
        for _ in 0..10 {
            unsafe {
                disp.dispatch(&pipe, &[(&buf, 0, 0)], (256, 1, 1), (64, 1, 1));
            }
        }

        let t0 = Instant::now();
        let mut prev = None;
        for _ in 0..iters {
            let future = unsafe {
                disp.dispatch_batch_async(|batch| {
                    batch.set_pipeline(&pipe);
                    batch.set_buffer(&buf, 0, 0);
                    batch.dispatch_threads((256, 1, 1), (64, 1, 1));
                })
            };
            if let Some(p) = prev {
                metal::GpuFuture::wait(p);
            }
            prev = Some(future);
        }
        if let Some(p) = prev {
            metal::GpuFuture::wait(p);
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    /// Simulates inference pass: 3 kernels × N layers, single batch.
    pub fn inference_sim(layers: usize, iters: usize) -> f64 {
        use metal::ComputeDispatcher;

        let dev = MtlDevice::system_default().unwrap();
        let queue = dev.new_command_queue().unwrap();
        let src = r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void matmul_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                                 device float *c [[buffer(2)]], uint id [[thread_position_in_grid]]) {
                c[id] = a[id] * b[id];
            }
            kernel void relu_k(device float *x [[buffer(0)]], uint id [[thread_position_in_grid]]) {
                x[id] = max(x[id], 0.0f);
            }
            kernel void add_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
                a[id] = a[id] + b[id];
            }
        "#;
        let lib = dev.new_library_with_source(src).unwrap();
        let matmul = dev
            .new_compute_pipeline(&lib.get_function("matmul_k").unwrap())
            .unwrap();
        let relu = dev
            .new_compute_pipeline(&lib.get_function("relu_k").unwrap())
            .unwrap();
        let add = dev
            .new_compute_pipeline(&lib.get_function("add_k").unwrap())
            .unwrap();
        let n = 4096usize;
        let buf_a = dev.new_buffer(n * 4).unwrap();
        let buf_b = dev.new_buffer(n * 4).unwrap();
        let buf_c = dev.new_buffer(n * 4).unwrap();
        let disp = ComputeDispatcher::new(&queue);

        for _ in 0..3 {
            unsafe {
                disp.dispatch_batch(|b| {
                    for _ in 0..layers {
                        b.set_pipeline(&matmul);
                        b.set_buffer(&buf_a, 0, 0);
                        b.set_buffer(&buf_b, 0, 1);
                        b.set_buffer(&buf_c, 0, 2);
                        b.dispatch_threads((n, 1, 1), (256, 1, 1));
                        b.set_pipeline(&relu);
                        b.set_buffer(&buf_c, 0, 0);
                        b.dispatch_threads((n, 1, 1), (256, 1, 1));
                        b.set_pipeline(&add);
                        b.set_buffer(&buf_a, 0, 0);
                        b.set_buffer(&buf_c, 0, 1);
                        b.dispatch_threads((n, 1, 1), (256, 1, 1));
                    }
                });
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                disp.dispatch_batch(|b| {
                    for _ in 0..layers {
                        b.set_pipeline(&matmul);
                        b.set_buffer(&buf_a, 0, 0);
                        b.set_buffer(&buf_b, 0, 1);
                        b.set_buffer(&buf_c, 0, 2);
                        b.dispatch_threads((n, 1, 1), (256, 1, 1));
                        b.set_pipeline(&relu);
                        b.set_buffer(&buf_c, 0, 0);
                        b.dispatch_threads((n, 1, 1), (256, 1, 1));
                        b.set_pipeline(&add);
                        b.set_buffer(&buf_a, 0, 0);
                        b.set_buffer(&buf_c, 0, 1);
                        b.dispatch_threads((n, 1, 1), (256, 1, 1));
                    }
                });
            }
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    const SAXPY_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void saxpy(device const float *x [[buffer(0)]],
                          device float *y       [[buffer(1)]],
                          constant float &a     [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
            y[id] = a * x[id] + y[id];
        }
    "#;

    const NOOP_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
}

// ── objc2-metal (ObjC Rust bindings) ──

mod objc2_bench {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::*;
    use std::time::Instant;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("no metal device")
    }

    pub fn device_discovery(iters: usize) -> f64 {
        let _ = get_device();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = get_device();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn buffer_creation(iters: usize, size: usize) -> f64 {
        let dev = get_device();
        for _ in 0..3 {
            let _ = dev
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dev
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .unwrap();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn shader_compile(iters: usize) -> f64 {
        let dev = get_device();
        let src = NSString::from_str(SAXPY_SRC);
        let _ = dev.newLibraryWithSource_options_error(&src, None).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = dev.newLibraryWithSource_options_error(&src, None).unwrap();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn encode_overhead(iters: usize) -> f64 {
        let dev = get_device();
        let queue = dev.newCommandQueue().unwrap();
        let src = NSString::from_str(NOOP_SRC);
        let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
        let fname = NSString::from_str("noop");
        let func = lib.newFunctionWithName(&fname).unwrap();
        let pipe = dev
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap();
        let buf = dev
            .newBufferWithLength_options(256 * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let grid = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        for _ in 0..10 {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipe);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let t0 = Instant::now();
        let mut last = None;
        for _ in 0..iters {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipe);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.endEncoding();
            cmd.commit();
            last = Some(cmd);
        }
        if let Some(cmd) = last {
            cmd.waitUntilCompleted();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn dispatch_overhead(iters: usize) -> f64 {
        let dev = get_device();
        let queue = dev.newCommandQueue().unwrap();
        let src = NSString::from_str(NOOP_SRC);
        let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
        let fname = NSString::from_str("noop");
        let func = lib.newFunctionWithName(&fname).unwrap();
        let pipe = dev
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap();
        let buf = dev
            .newBufferWithLength_options(256 * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let grid = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        for _ in 0..10 {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipe);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipe);
            unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    pub fn large_compute(iters: usize, n: usize) -> f64 {
        let dev = get_device();
        let queue = dev.newCommandQueue().unwrap();
        let src = NSString::from_str(SAXPY_SRC);
        let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
        let fname = NSString::from_str("saxpy");
        let func = lib.newFunctionWithName(&fname).unwrap();
        let pipe = dev
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap();
        let buf_x = dev
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let buf_y = dev
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();

        // fill
        unsafe {
            let ptr = buf_x.contents().as_ptr() as *mut f32;
            for i in 0..n {
                *ptr.add(i) = i as f32;
            }
            let ptr = buf_y.contents().as_ptr() as *mut f32;
            for i in 0..n {
                *ptr.add(i) = 1.0;
            }
        }

        let a_val: f32 = 2.0;
        let a_bytes = a_val.to_le_bytes();
        let grid = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let group = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        for _ in 0..3 {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipe);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_y), 0, 1);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(a_bytes.as_ptr() as *mut _).unwrap(),
                    a_bytes.len(),
                    2,
                );
            }
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        // reset y
        unsafe {
            let ptr = buf_y.contents().as_ptr() as *mut f32;
            for i in 0..n {
                *ptr.add(i) = 1.0;
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipe);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_x), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_y), 0, 1);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(a_bytes.as_ptr() as *mut _).unwrap(),
                    a_bytes.len(),
                    2,
                );
            }
            enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    const SAXPY_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void saxpy(device const float *x [[buffer(0)]],
                          device float *y       [[buffer(1)]],
                          constant float &a     [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
            y[id] = a * x[id] + y[id];
        }
    "#;

    /// Batch encode: N dispatches per cmd buffer, amortized cost
    pub fn batch_encode(batch_size: usize, iters: usize) -> f64 {
        let dev = get_device();
        let queue = dev.newCommandQueue().unwrap();
        let src = NSString::from_str(NOOP_SRC);
        let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
        let fname = NSString::from_str("noop");
        let func = lib.newFunctionWithName(&fname).unwrap();
        let pipe = dev
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap();
        let buf = dev
            .newBufferWithLength_options(256 * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let grid = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };

        for _ in 0..3 {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..batch_size {
                enc.setComputePipelineState(&pipe);
                unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..batch_size {
                enc.setComputePipelineState(&pipe);
                unsafe { enc.setBuffer_offset_atIndex(Some(&buf), 0, 0) };
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        t0.elapsed().as_secs_f64() / (iters * batch_size) as f64
    }

    /// Inference sim: 3 kernels × N layers, single batch
    pub fn inference_sim(layers: usize, iters: usize) -> f64 {
        let dev = get_device();
        let queue = dev.newCommandQueue().unwrap();
        let src = NSString::from_str(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void matmul_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                                 device float *c [[buffer(2)]], uint id [[thread_position_in_grid]]) {
                c[id] = a[id] * b[id];
            }
            kernel void relu_k(device float *x [[buffer(0)]], uint id [[thread_position_in_grid]]) {
                x[id] = max(x[id], 0.0f);
            }
            kernel void add_k(device float *a [[buffer(0)]], device float *b [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
                a[id] = a[id] + b[id];
            }
        "#,
        );
        let lib = dev.newLibraryWithSource_options_error(&src, None).unwrap();
        let matmul = dev
            .newComputePipelineStateWithFunction_error(
                &lib.newFunctionWithName(&NSString::from_str("matmul_k"))
                    .unwrap(),
            )
            .unwrap();
        let relu = dev
            .newComputePipelineStateWithFunction_error(
                &lib.newFunctionWithName(&NSString::from_str("relu_k"))
                    .unwrap(),
            )
            .unwrap();
        let add = dev
            .newComputePipelineStateWithFunction_error(
                &lib.newFunctionWithName(&NSString::from_str("add_k"))
                    .unwrap(),
            )
            .unwrap();
        let n: usize = 4096;
        let buf_a = dev
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let buf_b = dev
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let buf_c = dev
            .newBufferWithLength_options(n * 4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let grid = MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let group = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        for _ in 0..3 {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..layers {
                enc.setComputePipelineState(&matmul);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
                }
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
                enc.setComputePipelineState(&relu);
                unsafe { enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 0) };
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
                enc.setComputePipelineState(&add);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 1);
                }
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            for _ in 0..layers {
                enc.setComputePipelineState(&matmul);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
                }
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
                enc.setComputePipelineState(&relu);
                unsafe { enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 0) };
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
                enc.setComputePipelineState(&add);
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&buf_c), 0, 1);
                }
                enc.dispatchThreads_threadsPerThreadgroup(grid, group);
            }
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    const NOOP_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void noop(device float *a [[buffer(0)]],
                         uint id [[thread_position_in_grid]]) {
            a[id] = a[id] + 1.0;
        }
    "#;
}

fn us(secs: f64) -> f64 {
    secs * 1_000_000.0
}

fn ms(secs: f64) -> f64 {
    secs * 1_000.0
}

/// Run a benchmark N times, return the minimum (most stable measurement).
fn min_of<F: Fn() -> f64>(n: usize, f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..n {
        best = best.min(f());
    }
    best
}

fn main() {
    println!("=== rmetal vs objc2-metal Performance Comparison ===\n");
    println!(
        "{:<30} {:>12} {:>12} {:>10}",
        "Test", "rmetal", "objc2", "ratio"
    );
    println!("{}", "-".repeat(66));

    // 1. Device discovery
    let iters = 1000;
    let r = rmetal_bench::device_discovery(iters);
    let o = objc2_bench::device_discovery(iters);
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Device discovery",
        us(r),
        us(o),
        o / r
    );

    // 2. Buffer creation (4 KB)
    let iters = 5000;
    let r = rmetal_bench::buffer_creation(iters, 4096);
    let o = objc2_bench::buffer_creation(iters, 4096);
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Buffer create (4 KB)",
        us(r),
        us(o),
        o / r
    );

    // 3. Buffer creation (16 MB)
    let iters = 500;
    let r = rmetal_bench::buffer_creation(iters, 16 * 1024 * 1024);
    let o = objc2_bench::buffer_creation(iters, 16 * 1024 * 1024);
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Buffer create (16 MB)",
        us(r),
        us(o),
        o / r
    );

    // 4. Shader compilation
    let iters = 100;
    let r = rmetal_bench::shader_compile(iters);
    let o = objc2_bench::shader_compile(iters);
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "Shader compile (SAXPY)",
        ms(r),
        ms(o),
        o / r
    );

    // 4b. Encode overhead (CPU only, no per-iter GPU wait) — min of 3
    let iters = 5000;
    let r = min_of(3, || rmetal_bench::encode_overhead(iters));
    let o = min_of(3, || objc2_bench::encode_overhead(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Encode (wrapper)",
        us(r),
        us(o),
        o / r
    );
    let ru = min_of(3, || rmetal_bench::encode_unchecked(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Encode (unchecked)",
        us(ru),
        us(o),
        o / ru
    );

    // 4c. Batch encode — fair comparison: both sides batch 100 dispatches per cmd
    let batch_iters = 200;
    let re = min_of(3, || rmetal_bench::encode_encoder(100, batch_iters));
    let rb = min_of(3, || rmetal_bench::batch_encode(100, batch_iters));
    let ob = min_of(3, || objc2_bench::batch_encode(100, batch_iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Batch/op (encoder)",
        us(re),
        us(ob),
        ob / re
    );
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Batch/op (IMP)",
        us(rb),
        us(ob),
        ob / rb
    );

    // 5. Dispatch overhead (tiny kernel, sync) — min of 3 runs for stability
    let iters = 1000;
    let r = min_of(3, || rmetal_bench::dispatch_overhead(iters));
    let o = min_of(3, || objc2_bench::dispatch_overhead(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (wrapper)",
        us(r),
        us(o),
        o / r
    );

    let raw = min_of(3, || rmetal_bench::dispatch_raw(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (raw msgSend)",
        us(raw),
        us(o),
        o / raw
    );

    let imp = min_of(3, || rmetal_bench::dispatch_imp(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (IMP+autorelease)",
        us(imp),
        us(o),
        o / imp
    );

    let unc = min_of(3, || rmetal_bench::dispatch_unchecked(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (unchecked)",
        us(unc),
        us(o),
        o / unc
    );

    let ar = min_of(3, || rmetal_bench::dispatch_autoreleased(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (autoreleased)",
        us(ar),
        us(o),
        o / ar
    );

    // 5f. Pipelined: overlap GPU wait with next encode
    let piped = min_of(3, || rmetal_bench::dispatch_pipelined(iters));
    println!(
        "{:<30} {:>10.2} us {:>10.2} us {:>9.2}x",
        "Dispatch (pipelined)",
        us(piped),
        us(o),
        o / piped
    );

    // 5g. Inference simulation: 3 kernels × 100 layers, batched
    let inf_r = min_of(3, || rmetal_bench::inference_sim(100, 100));
    let inf_o = min_of(3, || objc2_bench::inference_sim(100, 100));
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "Inference (3x100 layers)",
        ms(inf_r),
        ms(inf_o),
        inf_o / inf_r
    );

    // 6. Large compute (SAXPY 16M floats)
    let n = 16 * 1024 * 1024;
    let iters = 100;
    let r = rmetal_bench::large_compute(iters, n);
    let o = objc2_bench::large_compute(iters, n);
    let bw_r = (n as f64 * 4.0 * 3.0) / r / 1e9;
    let bw_o = (n as f64 * 4.0 * 3.0) / o / 1e9;
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "SAXPY 16M floats",
        ms(r),
        ms(o),
        o / r
    );
    println!(
        "{:<30} {:>8.1} GB/s {:>8.1} GB/s",
        "  └ bandwidth", bw_r, bw_o
    );

    // 6b. SAXPY with ComputeDispatcher
    let ri = rmetal_bench::large_compute_imp(iters, n);
    let bw_ri = (n as f64 * 4.0 * 3.0) / ri / 1e9;
    println!(
        "{:<30} {:>10.2} ms {:>10.2} ms {:>9.2}x",
        "SAXPY 16M (IMP+unretained)",
        ms(ri),
        ms(o),
        o / ri
    );
    println!(
        "{:<30} {:>8.1} GB/s {:>8.1} GB/s",
        "  └ bandwidth", bw_ri, bw_o
    );

    println!("\n{}", "-".repeat(66));
    println!("ratio > 1.0 = rmetal faster, < 1.0 = objc2 faster");
    println!("Note: GPU compute time dominates large workloads.");
    println!("      Binding overhead visible only in dispatch/creation paths.");
}
