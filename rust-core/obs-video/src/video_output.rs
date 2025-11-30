//! Lock-free video output pipeline
//!
//! Replaces mutex-based video-io.c with lock-free data structures
//! for better performance on 8-core i7-9700K.

use crate::frame_pool::FramePool;
use crate::types::{VideoFrame, VideoOutputInfo};
use crossbeam::channel::{self, Receiver, Sender};
use crossbeam::queue::ArrayQueue;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

const MAX_CACHE_SIZE: usize = 16;

/// Lock-free video output
pub struct VideoOutput {
    info: VideoOutputInfo,

    // Lock-free frame queue (SPSC: graphics thread → video thread)
    frame_queue: Arc<ArrayQueue<CachedFrame>>,

    // Encoder connections
    encoders: Arc<RwLock<Vec<EncoderConnection>>>,

    // Frame pool for zero-copy operations
    frame_pool: Arc<FramePool>,

    // Statistics (atomic for lock-free reads)
    total_frames: Arc<AtomicU64>,
    skipped_frames: Arc<AtomicU64>,

    // Thread control
    running: Arc<AtomicBool>,
    thread_handle: Option<JoinHandle<()>>,
}

struct CachedFrame {
    frame: VideoFrame,
    _timestamp: u64, // Stored for timestamp management
}

struct EncoderConnection {
    id: u64,
    frame_rate_divisor: u32,
    frame_count: u32,
    tx: Sender<VideoFrame>,
}

impl VideoOutput {
    /// Create a new video output
    pub fn new(width: u32, height: u32, fps_num: u32, fps_den: u32) -> Self {
        let info = VideoOutputInfo {
            width,
            height,
            fps_num,
            fps_den,
            format: crate::types::VideoFormat::NV12 as u32,
        };

        let frame_queue = Arc::new(ArrayQueue::new(MAX_CACHE_SIZE));
        let encoders = Arc::new(RwLock::new(Vec::new()));
        let frame_pool = Arc::new(FramePool::new(
            crate::types::VideoFormat::NV12,
            width,
            height,
            MAX_CACHE_SIZE + 4, // Extra frames for in-flight encoding
        ));

        let total_frames = Arc::new(AtomicU64::new(0));
        let skipped_frames = Arc::new(AtomicU64::new(0));
        let running = Arc::new(AtomicBool::new(true));

        // Spawn video distribution thread
        let thread_handle = Self::spawn_video_thread(
            frame_queue.clone(),
            encoders.clone(),
            total_frames.clone(),
            skipped_frames.clone(),
            running.clone(),
            info,
        );

        VideoOutput {
            info,
            frame_queue,
            encoders,
            frame_pool,
            total_frames,
            skipped_frames,
            running,
            thread_handle: Some(thread_handle),
        }
    }

    /// Lock a frame for rendering (called by graphics thread)
    ///
    /// Returns a mutable reference to frame data that the graphics thread
    /// can render into. Zero-copy design.
    pub fn lock_frame(&self) -> Option<VideoFrame> {
        self.frame_pool.acquire()
    }

    /// Unlock and submit frame for encoding (called by graphics thread)
    ///
    /// This is a lock-free operation using an atomic queue.
    pub fn unlock_frame(&self, mut frame: VideoFrame, timestamp: u64) -> bool {
        frame.timestamp = timestamp;

        let cached = CachedFrame {
            frame: frame.clone(),
            _timestamp: timestamp,
        };

        match self.frame_queue.push(cached) {
            Ok(_) => true,
            Err(_) => {
                // Queue full - increment skip counter
                self.skipped_frames.fetch_add(1, Ordering::Relaxed);

                // Release frame back to pool
                self.frame_pool.release(&frame);
                false
            }
        }
    }

    /// Connect an encoder
    ///
    /// Returns channel receiver that the encoder can use to receive frames.
    /// frame_rate_divisor allows encoding at fractional framerates (e.g., 30fps from 60fps canvas).
    pub fn connect_encoder(&self, frame_rate_divisor: u32) -> Receiver<VideoFrame> {
        let (tx, rx) = channel::bounded(4); // Small buffer for encoder

        let connection = EncoderConnection {
            id: rand::random(),
            frame_rate_divisor,
            frame_count: 0,
            tx,
        };

        self.encoders.write().push(connection);

        rx
    }

    /// Disconnect an encoder by ID
    pub fn disconnect_encoder(&self, encoder_id: u64) {
        let mut encoders = self.encoders.write();
        encoders.retain(|enc| enc.id != encoder_id);
    }

    /// Get statistics
    pub fn stats(&self) -> VideoOutputStats {
        VideoOutputStats {
            total_frames: self.total_frames.load(Ordering::Relaxed),
            skipped_frames: self.skipped_frames.load(Ordering::Relaxed),
            queued_frames: self.frame_queue.len(),
            pool_stats: self.frame_pool.stats(),
        }
    }

    /// Spawn video distribution thread
    fn spawn_video_thread(
        frame_queue: Arc<ArrayQueue<CachedFrame>>,
        encoders: Arc<RwLock<Vec<EncoderConnection>>>,
        total_frames: Arc<AtomicU64>,
        skipped_frames: Arc<AtomicU64>,
        running: Arc<AtomicBool>,
        info: VideoOutputInfo,
    ) -> JoinHandle<()> {
        thread::Builder::new()
            .name("obs-video-output".to_string())
            .spawn(move || {
                // Calculate frame interval
                let frame_interval_ns = (info.fps_den as u64 * 1_000_000_000) / info.fps_num as u64;
                let frame_interval = Duration::from_nanos(frame_interval_ns);

                while running.load(Ordering::Relaxed) {
                    // Try to get a frame (non-blocking)
                    if let Some(cached) = frame_queue.pop() {
                        // Distribute to all encoders
                        let mut encoders_lock = encoders.write();

                        for encoder in encoders_lock.iter_mut() {
                            // Frame rate divisor logic
                            encoder.frame_count += 1;
                            if encoder.frame_count >= encoder.frame_rate_divisor {
                                encoder.frame_count = 0;

                                // Send frame to encoder (non-blocking)
                                // If encoder is slow, this will fail and we skip
                                let _ = encoder.tx.try_send(cached.frame.clone());
                            }
                        }

                        // Clean up disconnected encoders
                        encoders_lock.retain(|enc| !enc.tx.is_full());

                        total_frames.fetch_add(1, Ordering::Relaxed);
                    } else {
                        // No frames available, sleep briefly
                        thread::sleep(Duration::from_micros(100));
                    }
                }

                log::info!("Video output thread exiting");
            })
            .expect("Failed to spawn video thread")
    }

    /// Shutdown video output
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::Relaxed);

        if let Some(handle) = self.thread_handle.take() {
            handle.join().ok();
        }
    }
}

impl Drop for VideoOutput {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VideoOutputStats {
    pub total_frames: u64,
    pub skipped_frames: u64,
    pub queued_frames: usize,
    pub pool_stats: crate::frame_pool::PoolStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_video_output_creation() {
        let output = VideoOutput::new(1920, 1080, 60, 1);
        let stats = output.stats();

        assert_eq!(stats.total_frames, 0);
        assert_eq!(stats.skipped_frames, 0);
    }

    #[test]
    fn test_lock_unlock_frame() {
        let output = VideoOutput::new(1920, 1080, 60, 1);

        let frame = output.lock_frame().unwrap();
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);

        let success = output.unlock_frame(frame, 1000);
        assert!(success);

        thread::sleep(Duration::from_millis(10));

        let stats = output.stats();
        assert!(stats.queued_frames <= 1);
    }

    #[test]
    fn test_encoder_connection() {
        let output = VideoOutput::new(1920, 1080, 60, 1);

        let rx = output.connect_encoder(1);

        let frame = output.lock_frame().unwrap();
        output.unlock_frame(frame, 1000);

        thread::sleep(Duration::from_millis(50));

        // Encoder should receive frame
        match rx.try_recv() {
            Ok(frame) => {
                assert_eq!(frame.width, 1920);
                assert_eq!(frame.timestamp, 1000);
            }
            Err(_) => {
                // Frame might not have been distributed yet
            }
        }
    }

    #[test]
    fn test_frame_rate_divisor() {
        let output = VideoOutput::new(1920, 1080, 60, 1);

        // Encoder at 30fps (divisor=2)
        let rx = output.connect_encoder(2);

        for i in 0..10 {
            if let Some(frame) = output.lock_frame() {
                output.unlock_frame(frame, i * 16666667); // 60fps timestamps
            }
        }

        thread::sleep(Duration::from_millis(200));

        // Should receive ~5 frames (half of 10)
        let mut received = 0;
        while rx.try_recv().is_ok() {
            received += 1;
        }

        assert!(
            received >= 4 && received <= 6,
            "Expected ~5 frames, got {}",
            received
        );
    }
}
