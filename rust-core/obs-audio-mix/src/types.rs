//! Audio data types

use std::fmt;

/// Audio sample format
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Unknown = 0,
    U8Bit = 1,      // Unsigned 8-bit
    I16Bit = 2,     // Signed 16-bit
    I32Bit = 3,     // Signed 32-bit
    Float32Bit = 4, // 32-bit float (OBS default)
    U8BitPlanar = 5,
    I16BitPlanar = 6,
    I32BitPlanar = 7,
    Float32Planar = 8, // Planar float (most efficient for SIMD)
}

impl AudioFormat {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            AudioFormat::U8Bit | AudioFormat::U8BitPlanar => 1,
            AudioFormat::I16Bit | AudioFormat::I16BitPlanar => 2,
            AudioFormat::I32Bit | AudioFormat::I32BitPlanar => 4,
            AudioFormat::Float32Bit | AudioFormat::Float32Planar => 4,
            AudioFormat::Unknown => 0,
        }
    }

    pub fn is_planar(self) -> bool {
        matches!(
            self,
            AudioFormat::U8BitPlanar
                | AudioFormat::I16BitPlanar
                | AudioFormat::I32BitPlanar
                | AudioFormat::Float32Planar
        )
    }
}

/// Audio speaker layout
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerLayout {
    Unknown = 0,
    Mono = 1,
    Stereo = 2,
    TwoOne = 3,   // 2.1
    Quad = 4,     // 4.0
    FourOne = 5,  // 4.1
    FiveOne = 6,  // 5.1
    SevenOne = 8, // 7.1 (yes, 8 not 7 - it's channel count)
}

impl SpeakerLayout {
    pub fn channel_count(self) -> usize {
        match self {
            SpeakerLayout::Unknown => 0,
            SpeakerLayout::Mono => 1,
            SpeakerLayout::Stereo => 2,
            SpeakerLayout::TwoOne => 3,
            SpeakerLayout::Quad => 4,
            SpeakerLayout::FourOne => 5,
            SpeakerLayout::FiveOne => 6,
            SpeakerLayout::SevenOne => 8,
        }
    }
}

/// Audio buffer configuration
#[derive(Debug, Clone, Copy)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
    pub format: AudioFormat,
    pub layout: SpeakerLayout,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            frames: 1024,
            format: AudioFormat::Float32Planar,
            layout: SpeakerLayout::Stereo,
        }
    }
}

/// Audio data buffer
pub struct AudioData {
    pub data: Vec<Vec<f32>>, // Per-channel planar data
    pub frames: usize,
    pub timestamp: u64,
}

impl AudioData {
    pub fn new(channels: usize, frames: usize) -> Self {
        let mut data = Vec::with_capacity(channels);
        for _ in 0..channels {
            data.push(vec![0.0f32; frames]);
        }

        Self {
            data,
            frames,
            timestamp: 0,
        }
    }

    pub fn channels(&self) -> usize {
        self.data.len()
    }

    pub fn clear(&mut self) {
        for channel in &mut self.data {
            channel.fill(0.0);
        }
    }
}

impl fmt::Debug for AudioData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioData")
            .field("channels", &self.channels())
            .field("frames", &self.frames)
            .field("timestamp", &self.timestamp)
            .finish()
    }
}
