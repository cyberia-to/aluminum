//! MtlLibrary + MtlFunction: shader compilation from Metal Shading Language

use crate::ffi::*;
use crate::MetalError;

/// A compiled shader library. Wraps `id<MTLLibrary>`.
pub struct MtlLibrary {
    raw: ObjcId,
}

impl MtlLibrary {
    pub(crate) fn from_raw(raw: ObjcId) -> Self {
        MtlLibrary { raw }
    }

    /// Get a function by name from the library.
    pub fn get_function(&self, name: &str) -> Result<MtlFunction, MetalError> {
        let ns_name = nsstring(name);
        let raw = unsafe {
            let r = msg1::<ObjcId>(self.raw, SEL_newFunctionWithName(), ns_name);
            CFRelease(ns_name as CFTypeRef);
            r
        };
        if raw.is_null() {
            return Err(MetalError::FunctionNotFound(name.into()));
        }
        Ok(MtlFunction { raw })
    }

    /// List all function names in the library.
    pub fn function_names(&self) -> Vec<String> {
        unsafe {
            let arr = msg0(self.raw, SEL_functionNames());
            if arr.is_null() {
                return vec![];
            }
            let count = msg0_usize(arr, SEL_count());
            let mut names = Vec::with_capacity(count);
            for i in 0..count {
                let ns = msg1::<NSUInteger>(arr, SEL_objectAtIndex(), i);
                if let Some(s) = nsstring_to_rust(ns) {
                    names.push(s);
                }
            }
            names
        }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlLibrary {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// A single shader function. Wraps `id<MTLFunction>`.
pub struct MtlFunction {
    raw: ObjcId,
}

impl MtlFunction {
    pub fn name(&self) -> String {
        unsafe {
            let ns = msg0(self.raw, SEL_name());
            nsstring_to_rust(ns).unwrap_or_else(|| "unknown".into())
        }
    }

    pub fn as_raw(&self) -> ObjcId {
        self.raw
    }
}

impl Drop for MtlFunction {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}
