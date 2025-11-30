//! OBS Audio Mixer - Rust Optimization
//!
//! High-performance audio mixing with AVX SIMD optimization.
//!
//! Key features:
//! - AVX SIMD for audio clamping (6-8x speedup)
//! - Parallel mix processing with rayon
//! - Lock-free encoder dispatch

pub mod mixer;
pub mod clamping;
pub mod types;

pub use mixer::*;
pub use clamping::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_mix_creation() {
        let mix = AudioMix::new(48000, 1024, 8);
        assert_eq!(mix.sample_rate(), 48000);
        assert_eq!(mix.frames_per_buffer(), 1024);
        assert_eq!(mix.channels(), 8);
    }
}
