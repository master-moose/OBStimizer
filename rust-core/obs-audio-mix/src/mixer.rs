//! Audio mixer implementation with parallel processing

use crate::clamping::*;
use crate::types::*;
use crossbeam::channel::{self, Receiver, Sender};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

const MAX_AUDIO_MIXES: usize = 6;

/// Audio mixer for multiple sources
pub struct AudioMixer {
    _config: AudioConfig, // Stored for future use (runtime reconfiguration)
    mixes: Vec<Arc<RwLock<AudioMix>>>,
    running: Arc<AtomicBool>,
    frames_processed: Arc<AtomicU64>,
}

impl AudioMixer {
    /// Create a new audio mixer
    pub fn new(config: AudioConfig) -> Self {
        let mut mixes = Vec::with_capacity(MAX_AUDIO_MIXES);
        for _ in 0..MAX_AUDIO_MIXES {
            mixes.push(Arc::new(RwLock::new(AudioMix::new(
                config.sample_rate,
                config.frames,
                config.channels,
            ))));
        }

        Self {
            _config: config,
            mixes,
            running: Arc::new(AtomicBool::new(true)),
            frames_processed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a specific mix
    pub fn get_mix(&self, index: usize) -> Option<Arc<RwLock<AudioMix>>> {
        if index < self.mixes.len() {
            Some(self.mixes[index].clone())
        } else {
            None
        }
    }

    /// Process all active mixes in parallel
    pub fn process_all_mixes(&self) {
        use rayon::prelude::*;

        // Parallel processing of all mixes
        self.mixes.par_iter().for_each(|mix| {
            let mut mix_guard = mix.write();
            if mix_guard.has_inputs() {
                mix_guard.process();
            }
        });

        self.frames_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get mixer statistics
    pub fn stats(&self) -> AudioMixerStats {
        let mut total_inputs = 0;
        let mut active_mixes = 0;

        for mix in &self.mixes {
            let mix_guard = mix.read();
            if mix_guard.has_inputs() {
                active_mixes += 1;
                total_inputs += mix_guard.input_count();
            }
        }

        AudioMixerStats {
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            active_mixes,
            total_inputs,
        }
    }

    /// Shutdown the mixer
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Individual audio mix
pub struct AudioMix {
    sample_rate: u32,
    frames_per_buffer: usize,
    channels: usize,

    // Mix buffers (clamped and unclamped versions)
    buffer_clamped: AudioData,
    buffer_unclamped: AudioData,

    // Encoder connections
    encoders: Vec<EncoderConnection>,
}

struct EncoderConnection {
    tx: Sender<AudioData>,
    use_unclamped: bool,
}

impl AudioMix {
    /// Create a new audio mix
    pub fn new(sample_rate: u32, frames_per_buffer: usize, channels: usize) -> Self {
        Self {
            sample_rate,
            frames_per_buffer,
            channels,
            buffer_clamped: AudioData::new(channels, frames_per_buffer),
            buffer_unclamped: AudioData::new(channels, frames_per_buffer),
            encoders: Vec::new(),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn frames_per_buffer(&self) -> usize {
        self.frames_per_buffer
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn has_inputs(&self) -> bool {
        !self.encoders.is_empty()
    }

    pub fn input_count(&self) -> usize {
        self.encoders.len()
    }

    /// Connect an encoder to this mix
    ///
    /// Returns a receiver channel for the encoder to receive audio data.
    pub fn connect_encoder(&mut self, use_unclamped: bool) -> Receiver<AudioData> {
        let (tx, rx) = channel::bounded(4);

        self.encoders.push(EncoderConnection { tx, use_unclamped });

        rx
    }

    /// Disconnect all encoders
    pub fn disconnect_all(&mut self) {
        self.encoders.clear();
    }

    /// Mix audio sources into the buffer
    ///
    /// In a real implementation, this would receive audio from multiple sources
    /// and mix them together. For now, we'll demonstrate the clamping optimization.
    pub fn mix_sources(&mut self, sources: &[AudioData]) {
        // Clear buffers
        self.buffer_clamped.clear();
        self.buffer_unclamped.clear();

        // Mix all sources (simple addition)
        for source in sources {
            for (ch_idx, channel) in source.data.iter().enumerate() {
                if ch_idx >= self.channels {
                    break;
                }

                for (frame_idx, sample) in channel.iter().enumerate() {
                    if frame_idx >= self.frames_per_buffer {
                        break;
                    }

                    self.buffer_unclamped.data[ch_idx][frame_idx] += sample;
                }
            }
        }

        // Copy unclamped to clamped buffer
        for (ch_idx, channel) in self.buffer_unclamped.data.iter().enumerate() {
            self.buffer_clamped.data[ch_idx].copy_from_slice(channel);
        }

        // Clamp the clamped buffer using AVX SIMD
        clamp_audio_channels(&mut self.buffer_clamped.data);
    }

    /// Process this mix and dispatch to encoders
    pub fn process(&mut self) {
        // In a real implementation, sources would be passed in
        // For now, we'll just demonstrate encoder dispatch

        // Dispatch to encoders (disconnected ones will fail to send)
        for encoder in &self.encoders {
            let data = if encoder.use_unclamped {
                // Send unclamped data for encoders that support it
                self.buffer_unclamped.clone()
            } else {
                // Send clamped data for safety
                self.buffer_clamped.clone()
            };

            // Non-blocking send (ignore errors from disconnected channels)
            let _ = encoder.tx.try_send(data);
        }

        // Clean up disconnected encoders
        self.encoders.retain(|enc| !enc.tx.is_full());
    }
}

impl Clone for AudioData {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            frames: self.frames,
            timestamp: self.timestamp,
        }
    }
}

/// Mixer statistics
#[derive(Debug, Clone, Copy)]
pub struct AudioMixerStats {
    pub frames_processed: u64,
    pub active_mixes: usize,
    pub total_inputs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_mixer_creation() {
        let config = AudioConfig::default();
        let mixer = AudioMixer::new(config);

        let stats = mixer.stats();
        assert_eq!(stats.frames_processed, 0);
        assert_eq!(stats.active_mixes, 0);
    }

    #[test]
    fn test_audio_mix() {
        let mut mix = AudioMix::new(48000, 1024, 2);

        assert_eq!(mix.sample_rate(), 48000);
        assert_eq!(mix.frames_per_buffer(), 1024);
        assert_eq!(mix.channels(), 2);
        assert!(!mix.has_inputs());
    }

    #[test]
    fn test_encoder_connection() {
        let mut mix = AudioMix::new(48000, 1024, 2);

        let rx = mix.connect_encoder(false);
        assert!(mix.has_inputs());
        assert_eq!(mix.input_count(), 1);

        // Keep rx alive - disconnection happens when encoder drops
        assert!(rx.is_empty()); // Channel should be empty initially
    }

    #[test]
    fn test_mix_sources() {
        let mut mix = AudioMix::new(48000, 1024, 2);

        // Create two test sources
        let mut source1 = AudioData::new(2, 1024);
        let mut source2 = AudioData::new(2, 1024);

        // Fill with test data
        for channel in &mut source1.data {
            for sample in channel.iter_mut() {
                *sample = 0.5;
            }
        }

        for channel in &mut source2.data {
            for sample in channel.iter_mut() {
                *sample = 0.3;
            }
        }

        mix.mix_sources(&[source1, source2]);

        // Result should be 0.5 + 0.3 = 0.8 (clamped)
        assert!((mix.buffer_clamped.data[0][0] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_mix_clamping() {
        let mut mix = AudioMix::new(48000, 1024, 2);

        // Create source with out-of-range values
        let mut source = AudioData::new(2, 1024);
        for channel in &mut source.data {
            for sample in channel.iter_mut() {
                *sample = 1.5; // Out of range
            }
        }

        mix.mix_sources(&[source]);

        // Clamped buffer should be 1.0
        assert_eq!(mix.buffer_clamped.data[0][0], 1.0);

        // Unclamped buffer should still be 1.5
        assert_eq!(mix.buffer_unclamped.data[0][0], 1.5);
    }

    #[test]
    fn test_parallel_processing() {
        let config = AudioConfig::default();
        let mixer = AudioMixer::new(config);

        // Connect encoders to multiple mixes
        for i in 0..3 {
            let mix_arc = mixer.get_mix(i).unwrap();
            let mut mix = mix_arc.write();
            mix.connect_encoder(false);
        }

        // Process all mixes in parallel
        mixer.process_all_mixes();

        let stats = mixer.stats();
        assert_eq!(stats.active_mixes, 3);
        assert_eq!(stats.frames_processed, 1);
    }
}
