//! aluminium driver benchmarks — measures Metal.framework overhead

mod basic;
mod dispatch;
mod encode;

pub use basic::*;
pub use dispatch::*;
pub use encode::*;

pub const SAXPY_SRC: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void saxpy(device const float *x [[buffer(0)]],
                      device float *y       [[buffer(1)]],
                      constant float &a     [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
        y[id] = a * x[id] + y[id];
    }
"#;

pub const NOOP_SRC: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void noop(device float *a [[buffer(0)]],
                     uint id [[thread_position_in_grid]]) {
        a[id] = a[id] + 1.0;
    }
"#;
