//! OBS Scene Compositor - Rust Optimization
//!
//! High-performance scene rendering optimized for i7-9700K.
//!
//! Key optimizations:
//! - Array-based storage (cache-friendly, replaces linked list)
//! - Lock-free rendering with RwLock (concurrent reads)
//! - SIMD matrix operations via glam
//! - Dirty flag system for transform caching

pub mod render;
pub mod scene;
pub mod transform;
pub mod types;

pub use render::*;
pub use scene::*;
pub use transform::*;
pub use types::*;

pub fn version() -> &'static str {
    "0.1.0"
}
