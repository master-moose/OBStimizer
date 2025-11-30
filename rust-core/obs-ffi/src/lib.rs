//! OBS FFI - C Foreign Function Interface
//!
//! Provides C-compatible API for integration with existing OBS codebase.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;

// Re-export types from other crates
use obs_audio_mix::{AudioConfig, AudioFormat, AudioMixer, SpeakerLayout};
use obs_compositor::{Scene, SceneItem};
use obs_video::{VideoFormat, VideoFrame, VideoOutput, VideoOutputInfo};

mod compositor_ffi;
pub use compositor_ffi::*;

/// Opaque handle to VideoOutput (C-compatible)
pub struct OBSVideoOutput {
    _private: [u8; 0],
}

/// Opaque handle to AudioMixer (C-compatible)
pub struct OBSAudioMixer {
    _private: [u8; 0],
}

/// Opaque handle to Scene (C-compatible)
pub struct OBSScene {
    _private: [u8; 0],
}

/// C-compatible video frame structure
#[repr(C)]
pub struct CVideoFrame {
    pub data: [*mut u8; 4],
    pub linesize: [u32; 4],
    pub width: u32,
    pub height: u32,
    pub format: u32,
    pub timestamp: u64,
}

/// C-compatible video output info
#[repr(C)]
pub struct CVideoOutputInfo {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub format: u32,
}

/// C-compatible audio config
#[repr(C)]
pub struct CAudioConfig {
    pub sample_rate: u32,
    pub channels: u32,
    pub frames: u32,
    pub format: u32,
    pub layout: u32,
}

// ============================================================================
// VIDEO OUTPUT API
// ============================================================================

/// Create a new video output
///
/// # Safety
/// Caller must ensure parameters are valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_video_output_create(
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
) -> *mut OBSVideoOutput {
    let output = Box::new(VideoOutput::new(width, height, fps_num, fps_den));
    Box::into_raw(output) as *mut OBSVideoOutput
}

/// Destroy video output
///
/// # Safety
/// Caller must ensure ptr is valid and not already freed.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_video_output_destroy(ptr: *mut OBSVideoOutput) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut VideoOutput);
    }
}

/// Lock a frame for rendering
///
/// # Safety
/// Caller must ensure ptr is valid. frame_out must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_video_output_lock_frame(
    ptr: *mut OBSVideoOutput,
    frame_out: *mut CVideoFrame,
) -> c_int {
    if ptr.is_null() || frame_out.is_null() {
        return 0;
    }

    let output = &mut *(ptr as *mut VideoOutput);

    if let Some(frame) = output.lock_frame() {
        (*frame_out).data = frame.data;
        (*frame_out).linesize = frame.linesize;
        (*frame_out).width = frame.width;
        (*frame_out).height = frame.height;
        (*frame_out).format = frame.format as u32;
        (*frame_out).timestamp = frame.timestamp;
        1
    } else {
        0
    }
}

/// Unlock and submit frame for encoding
///
/// # Safety
/// Caller must ensure ptr and frame are valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_video_output_unlock_frame(
    ptr: *mut OBSVideoOutput,
    frame: *const CVideoFrame,
    timestamp: u64,
) -> c_int {
    if ptr.is_null() || frame.is_null() {
        return 0;
    }

    let output = &mut *(ptr as *mut VideoOutput);

    let rust_frame = VideoFrame {
        data: (*frame).data,
        linesize: (*frame).linesize,
        width: (*frame).width,
        height: (*frame).height,
        format: std::mem::transmute((*frame).format),
        timestamp: (*frame).timestamp,
    };

    if output.unlock_frame(rust_frame, timestamp) {
        1
    } else {
        0
    }
}

/// Get video output statistics
///
/// # Safety
/// Caller must ensure ptr is valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_video_output_get_total_frames(ptr: *const OBSVideoOutput) -> u64 {
    if ptr.is_null() {
        return 0;
    }

    let output = &*(ptr as *const VideoOutput);
    output.stats().total_frames
}

/// Get skipped frames count
///
/// # Safety
/// Caller must ensure ptr is valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_video_output_get_skipped_frames(
    ptr: *const OBSVideoOutput,
) -> u64 {
    if ptr.is_null() {
        return 0;
    }

    let output = &*(ptr as *const VideoOutput);
    output.stats().skipped_frames
}

// ============================================================================
// AUDIO MIXER API
// ============================================================================

/// Create a new audio mixer
///
/// # Safety
/// Caller must ensure config is valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_audio_mixer_create(
    config: *const CAudioConfig,
) -> *mut OBSAudioMixer {
    if config.is_null() {
        return ptr::null_mut();
    }

    let rust_config = AudioConfig {
        sample_rate: (*config).sample_rate,
        channels: (*config).channels as usize,
        frames: (*config).frames as usize,
        format: std::mem::transmute((*config).format),
        layout: std::mem::transmute((*config).layout),
    };

    let mixer = Box::new(AudioMixer::new(rust_config));
    Box::into_raw(mixer) as *mut OBSAudioMixer
}

/// Destroy audio mixer
///
/// # Safety
/// Caller must ensure ptr is valid and not already freed.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_audio_mixer_destroy(ptr: *mut OBSAudioMixer) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut AudioMixer);
    }
}

/// Process all audio mixes
///
/// # Safety
/// Caller must ensure ptr is valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_audio_mixer_process(ptr: *mut OBSAudioMixer) {
    if ptr.is_null() {
        return;
    }

    let mixer = &*(ptr as *const AudioMixer);
    mixer.process_all_mixes();
}

/// Get frames processed count
///
/// # Safety
/// Caller must ensure ptr is valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_audio_mixer_get_frames_processed(
    ptr: *const OBSAudioMixer,
) -> u64 {
    if ptr.is_null() {
        return 0;
    }

    let mixer = &*(ptr as *const AudioMixer);
    mixer.stats().frames_processed
}

// ============================================================================
// FORMAT CONVERSION API
// ============================================================================

/// Convert UYVY to NV12 using optimized SIMD
///
/// # Safety
/// Caller must ensure all pointers are valid and buffers are properly sized.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_convert_uyvy_to_nv12(
    input: *const u8,
    output_y: *mut u8,
    output_uv: *mut u8,
    width: u32,
    height: u32,
    in_linesize: u32,
    out_y_linesize: u32,
    out_uv_linesize: u32,
) -> c_int {
    if input.is_null() || output_y.is_null() || output_uv.is_null() {
        return 0;
    }

    let input_slice = std::slice::from_raw_parts(input, (height * in_linesize) as usize);
    let output_y_slice =
        std::slice::from_raw_parts_mut(output_y, (height * out_y_linesize) as usize);
    let output_uv_slice =
        std::slice::from_raw_parts_mut(output_uv, ((height / 2) * out_uv_linesize) as usize);

    obs_video::compress_uyvy_to_nv12(
        input_slice,
        output_y_slice,
        output_uv_slice,
        width as usize,
        height as usize,
        in_linesize as usize,
        out_y_linesize as usize,
        out_uv_linesize as usize,
    );

    1
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Get library version string
///
/// # Safety
/// Returns a static string, safe to call.
#[no_mangle]
pub extern "C" fn obs_rust_version() -> *const c_char {
    "0.1.0\0".as_ptr() as *const c_char
}

/// Check if AVX2 is available
///
/// # Safety
/// Safe to call, performs runtime CPU detection.
#[no_mangle]
pub extern "C" fn obs_rust_has_avx2() -> c_int {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return 1;
        }
    }
    0
}

/// Check if AVX is available
///
/// # Safety
/// Safe to call, performs runtime CPU detection.
#[no_mangle]
pub extern "C" fn obs_rust_has_avx() -> c_int {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            return 1;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_output_ffi() {
        unsafe {
            let output = obs_rust_video_output_create(1920, 1080, 60, 1);
            assert!(!output.is_null());

            let total = obs_rust_video_output_get_total_frames(output);
            assert_eq!(total, 0);

            obs_rust_video_output_destroy(output);
        }
    }

    #[test]
    fn test_audio_mixer_ffi() {
        unsafe {
            let config = CAudioConfig {
                sample_rate: 48000,
                channels: 2,
                frames: 1024,
                format: AudioFormat::Float32Planar as u32,
                layout: SpeakerLayout::Stereo as u32,
            };

            let mixer = obs_rust_audio_mixer_create(&config);
            assert!(!mixer.is_null());

            obs_rust_audio_mixer_process(mixer);

            let frames = obs_rust_audio_mixer_get_frames_processed(mixer);
            assert_eq!(frames, 1);

            obs_rust_audio_mixer_destroy(mixer);
        }
    }

    #[test]
    fn test_cpu_features() {
        let has_avx = obs_rust_has_avx();
        let has_avx2 = obs_rust_has_avx2();

        println!("AVX: {}, AVX2: {}", has_avx, has_avx2);
    }

    #[test]
    fn test_version() {
        unsafe {
            let version = obs_rust_version();
            assert!(!version.is_null());

            let version_str = CStr::from_ptr(version).to_str().unwrap();
            assert_eq!(version_str, "0.1.0");
        }
    }
}
