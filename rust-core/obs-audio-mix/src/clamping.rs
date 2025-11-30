//! Audio clamping with AVX SIMD optimization
//!
//! Clamps audio samples to [-1.0, 1.0] range and handles NaN values.
//! AVX implementation processes 8 floats per iteration (vs 1 for scalar).
//!
//! Expected performance: 6-8x speedup over scalar code.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Clamp audio buffer to [-1.0, 1.0] range with AVX SIMD
///
/// # Safety
/// Requires AVX CPU support. Buffer must be properly sized.
#[target_feature(enable = "avx")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn clamp_audio_avx(buffer: &mut [f32]) {
    let min_val = _mm256_set1_ps(-1.0);
    let max_val = _mm256_set1_ps(1.0);

    let mut i = 0;
    let len = buffer.len();

    // Process 8 floats per iteration
    while i + 8 <= len {
        // Load 8 floats
        let mut val = _mm256_loadu_ps(buffer.as_ptr().add(i));

        // NaN → 0.0 (NaN != NaN comparison is false)
        let nan_mask = _mm256_cmp_ps(val, val, _CMP_EQ_OQ);
        val = _mm256_and_ps(val, nan_mask);

        // Clamp to [-1.0, 1.0]
        val = _mm256_min_ps(val, max_val);
        val = _mm256_max_ps(val, min_val);

        // Store back
        _mm256_storeu_ps(buffer.as_mut_ptr().add(i), val);
        i += 8;
    }

    // Handle remaining samples (< 8)
    while i < len {
        let val = buffer[i];
        if val.is_nan() {
            buffer[i] = 0.0;
        } else {
            buffer[i] = val.clamp(-1.0, 1.0);
        }
        i += 1;
    }
}

/// Scalar fallback for clamping (portable, slower)
pub fn clamp_audio_scalar(buffer: &mut [f32]) {
    for sample in buffer.iter_mut() {
        let val = *sample;
        *sample = if val.is_nan() {
            0.0
        } else {
            val.clamp(-1.0, 1.0)
        };
    }
}

/// Auto-dispatch clamping with runtime CPU detection
pub fn clamp_audio(buffer: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe {
                clamp_audio_avx(buffer);
            }
            return;
        }
    }

    // Fallback to scalar
    clamp_audio_scalar(buffer);
}

/// Clamp multiple channels in parallel using rayon
pub fn clamp_audio_channels(buffers: &mut [Vec<f32>]) {
    use rayon::prelude::*;

    buffers.par_iter_mut().for_each(|buffer| {
        clamp_audio(buffer);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_in_range() {
        let mut buffer = vec![0.5, -0.5, 0.0, 0.9, -0.9];
        let expected = buffer.clone();

        clamp_audio(&mut buffer);

        assert_eq!(buffer, expected, "In-range values should not change");
    }

    #[test]
    fn test_clamp_out_of_range() {
        let mut buffer = vec![1.5, -1.5, 2.0, -2.0];
        clamp_audio(&mut buffer);

        assert_eq!(buffer[0], 1.0, "1.5 should clamp to 1.0");
        assert_eq!(buffer[1], -1.0, "-1.5 should clamp to -1.0");
        assert_eq!(buffer[2], 1.0, "2.0 should clamp to 1.0");
        assert_eq!(buffer[3], -1.0, "-2.0 should clamp to -1.0");
    }

    #[test]
    fn test_clamp_nan() {
        let mut buffer = vec![f32::NAN, 0.5, f32::NAN, -0.5];
        clamp_audio(&mut buffer);

        assert_eq!(buffer[0], 0.0, "NaN should become 0.0");
        assert_eq!(buffer[1], 0.5, "Valid value unchanged");
        assert_eq!(buffer[2], 0.0, "NaN should become 0.0");
        assert_eq!(buffer[3], -0.5, "Valid value unchanged");
    }

    #[test]
    fn test_clamp_large_buffer() {
        let mut buffer = vec![0.0f32; 10000];

        // Fill with test pattern
        for (i, sample) in buffer.iter_mut().enumerate() {
            *sample = ((i as f32) / 100.0).sin() * 1.5; // Range: -1.5 to 1.5
        }

        clamp_audio(&mut buffer);

        // All values should be in [-1.0, 1.0]
        for sample in &buffer {
            assert!(
                *sample >= -1.0 && *sample <= 1.0,
                "Sample {} out of range",
                sample
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx_vs_scalar() {
        if !is_x86_feature_detected!("avx") {
            return; // Skip if no AVX support
        }

        let mut buffer_avx = vec![0.0f32; 1024];
        let mut buffer_scalar = vec![0.0f32; 1024];

        // Fill with test pattern
        for i in 0..1024 {
            let val = ((i as f32) / 10.0).sin() * 2.0; // Range: -2.0 to 2.0
            buffer_avx[i] = val;
            buffer_scalar[i] = val;
        }

        // Add some NaN values
        buffer_avx[100] = f32::NAN;
        buffer_scalar[100] = f32::NAN;
        buffer_avx[500] = f32::NAN;
        buffer_scalar[500] = f32::NAN;

        unsafe {
            clamp_audio_avx(&mut buffer_avx);
        }
        clamp_audio_scalar(&mut buffer_scalar);

        // Results should match
        for (i, (avx, scalar)) in buffer_avx.iter().zip(buffer_scalar.iter()).enumerate() {
            assert!(
                (avx - scalar).abs() < 0.0001,
                "Mismatch at index {}: AVX={}, Scalar={}",
                i,
                avx,
                scalar
            );
        }
    }

    #[test]
    fn test_parallel_clamping() {
        let mut channels = vec![
            vec![1.5, -1.5, 0.5],
            vec![2.0, -2.0, f32::NAN],
            vec![0.9, -0.9, 1.2],
        ];

        clamp_audio_channels(&mut channels);

        assert_eq!(channels[0], vec![1.0, -1.0, 0.5]);
        assert_eq!(channels[1], vec![1.0, -1.0, 0.0]);
        assert_eq!(channels[2], vec![0.9, -0.9, 1.0]);
    }
}
