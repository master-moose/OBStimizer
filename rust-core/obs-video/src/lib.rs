//! OBS Video Pipeline - Rust Optimization
//!
//! High-performance video frame processing optimized for Intel i7-9700K (AVX2).
//!
//! Key optimizations:
//! - Lock-free ring buffer for frame distribution
//! - AVX2 SIMD for colorspace conversion (2x faster than SSE2)
//! - Zero-copy frame handling where possible
//! - Memory pooling to reduce allocation churn

pub mod format_conversion;
pub mod video_output;
pub mod frame_pool;
pub mod types;

pub use format_conversion::*;
pub use video_output::*;
pub use frame_pool::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_format_sizes() {
        assert_eq!(VideoFormat::I420.plane_count(), 3);
        assert_eq!(VideoFormat::NV12.plane_count(), 2);
        assert_eq!(VideoFormat::RGBA.plane_count(), 1);
    }
}
