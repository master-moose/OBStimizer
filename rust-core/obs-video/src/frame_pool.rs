//! Memory-efficient frame pool with alignment for SIMD operations

use crate::types::{VideoFormat, VideoFrame};
use parking_lot::Mutex;
use std::alloc::{alloc_zeroed, dealloc, Layout};

const CACHE_LINE_SIZE: usize = 64;
const FRAME_ALIGNMENT: usize = 32; // AVX2 requires 32-byte alignment

/// Pool of pre-allocated video frames to eliminate allocation churn
pub struct FramePool {
    frames: Mutex<Vec<PooledFrame>>,
    format: VideoFormat,
    width: u32,
    height: u32,
    capacity: usize,
}

struct PooledFrame {
    data: [*mut u8; 4],
    linesize: [u32; 4],
    layout: Option<Layout>,
    in_use: bool,
}

unsafe impl Send for PooledFrame {}
unsafe impl Sync for PooledFrame {}

impl FramePool {
    /// Create a new frame pool
    ///
    /// # Arguments
    /// * `format` - Video format
    /// * `width` - Frame width (should be multiple of 8 for AVX2)
    /// * `height` - Frame height
    /// * `capacity` - Number of frames to pre-allocate
    pub fn new(format: VideoFormat, width: u32, height: u32, capacity: usize) -> Self {
        let mut frames = Vec::with_capacity(capacity);

        for _ in 0..capacity {
            frames.push(Self::allocate_frame(format, width, height));
        }

        FramePool {
            frames: Mutex::new(frames),
            format,
            width,
            height,
            capacity,
        }
    }

    /// Allocate a single aligned frame
    fn allocate_frame(format: VideoFormat, width: u32, height: u32) -> PooledFrame {
        let mut data = [std::ptr::null_mut(); 4];
        let mut linesize = [0u32; 4];
        let mut layout = None;

        match format {
            VideoFormat::I420 => {
                // Y plane
                let y_size = (width * height) as usize;
                let y_linesize = width;

                // U and V planes (half resolution)
                let uv_width = width / 2;
                let uv_height = height / 2;
                let uv_size = (uv_width * uv_height) as usize;
                let uv_linesize = uv_width;

                // Allocate contiguous memory for all planes
                let total_size = y_size + uv_size * 2;
                let align = FRAME_ALIGNMENT;

                unsafe {
                    let layout_obj = Layout::from_size_align_unchecked(total_size, align);
                    let ptr = alloc_zeroed(layout_obj);

                    data[0] = ptr;
                    data[1] = ptr.add(y_size);
                    data[2] = ptr.add(y_size + uv_size);

                    linesize[0] = y_linesize;
                    linesize[1] = uv_linesize;
                    linesize[2] = uv_linesize;

                    layout = Some(layout_obj);
                }
            }

            VideoFormat::NV12 => {
                // Y plane
                let y_size = (width * height) as usize;
                let y_linesize = width;

                // UV plane (interleaved, half resolution)
                let uv_width = width;
                let uv_height = height / 2;
                let uv_size = (uv_width * uv_height) as usize;
                let uv_linesize = uv_width;

                let total_size = y_size + uv_size;
                let align = FRAME_ALIGNMENT;

                unsafe {
                    let layout_obj = Layout::from_size_align_unchecked(total_size, align);
                    let ptr = alloc_zeroed(layout_obj);

                    data[0] = ptr;
                    data[1] = ptr.add(y_size);

                    linesize[0] = y_linesize;
                    linesize[1] = uv_linesize;

                    layout = Some(layout_obj);
                }
            }

            VideoFormat::RGBA | VideoFormat::BGRA | VideoFormat::BGRX => {
                let size = (width * height * 4) as usize;
                let align = FRAME_ALIGNMENT;

                unsafe {
                    let layout_obj = Layout::from_size_align_unchecked(size, align);
                    let ptr = alloc_zeroed(layout_obj);

                    data[0] = ptr;
                    linesize[0] = width * 4;

                    layout = Some(layout_obj);
                }
            }

            _ => {
                // Generic allocation for other formats
                let size = format.calculate_size(width, height);
                let align = FRAME_ALIGNMENT;

                unsafe {
                    let layout_obj = Layout::from_size_align_unchecked(size, align);
                    let ptr = alloc_zeroed(layout_obj);

                    data[0] = ptr;
                    linesize[0] = width * format.bytes_per_pixel() as u32;

                    layout = Some(layout_obj);
                }
            }
        }

        PooledFrame {
            data,
            linesize,
            layout,
            in_use: false,
        }
    }

    /// Acquire a frame from the pool
    pub fn acquire(&self) -> Option<VideoFrame> {
        let mut frames = self.frames.lock();

        for frame in frames.iter_mut() {
            if !frame.in_use {
                frame.in_use = true;
                return Some(VideoFrame {
                    data: frame.data,
                    linesize: frame.linesize,
                    width: self.width,
                    height: self.height,
                    format: self.format,
                    timestamp: 0,
                });
            }
        }

        None // Pool exhausted
    }

    /// Release a frame back to the pool
    pub fn release(&self, frame: &VideoFrame) {
        let mut frames = self.frames.lock();

        for pooled_frame in frames.iter_mut() {
            if pooled_frame.data[0] == frame.data[0] {
                pooled_frame.in_use = false;
                return;
            }
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let frames = self.frames.lock();
        let in_use = frames.iter().filter(|f| f.in_use).count();

        PoolStats {
            capacity: self.capacity,
            in_use,
            available: self.capacity - in_use,
        }
    }
}

impl Drop for FramePool {
    fn drop(&mut self) {
        let mut frames = self.frames.lock();

        for frame in frames.iter_mut() {
            if let Some(layout) = frame.layout {
                unsafe {
                    if !frame.data[0].is_null() {
                        dealloc(frame.data[0], layout);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub capacity: usize,
    pub in_use: usize,
    pub available: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_pool_acquire_release() {
        let pool = FramePool::new(VideoFormat::NV12, 1920, 1080, 4);

        let stats = pool.stats();
        assert_eq!(stats.capacity, 4);
        assert_eq!(stats.available, 4);

        let frame1 = pool.acquire().unwrap();
        assert_eq!(pool.stats().in_use, 1);

        let frame2 = pool.acquire().unwrap();
        assert_eq!(pool.stats().in_use, 2);

        pool.release(&frame1);
        assert_eq!(pool.stats().in_use, 1);

        pool.release(&frame2);
        assert_eq!(pool.stats().in_use, 0);
    }

    #[test]
    fn test_frame_pool_exhaustion() {
        let pool = FramePool::new(VideoFormat::I420, 640, 480, 2);

        let _f1 = pool.acquire().unwrap();
        let _f2 = pool.acquire().unwrap();
        let f3 = pool.acquire();

        assert!(f3.is_none(), "Pool should be exhausted");
    }

    #[test]
    fn test_frame_alignment() {
        let pool = FramePool::new(VideoFormat::NV12, 1920, 1080, 1);
        let frame = pool.acquire().unwrap();

        // Check 32-byte alignment for AVX2
        assert_eq!(frame.data[0] as usize % 32, 0, "Y plane not aligned");
        assert_eq!(frame.data[1] as usize % 32, 0, "UV plane not aligned");
    }
}
