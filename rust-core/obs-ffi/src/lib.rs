//! OBS FFI - C Foreign Function Interface
//!
//! Provides C-compatible API for integration with existing OBS codebase.
//!
//! TODO: Implement C FFI layer with:
//! - cbindgen header generation
//! - Safe wrapper around Rust components
//! - ABI compatibility testing

#![allow(dead_code)]

pub fn version() -> &'static str {
    "0.1.0"
}
