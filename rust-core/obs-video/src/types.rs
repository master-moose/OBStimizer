//! Video data types and constants

use bytemuck::{Pod, Zeroable};

/// Video format enumeration
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFormat {
    None = 0,
    I420 = 1,  // Planar YUV 4:2:0
    NV12 = 2,  // Semi-planar YUV 4:2:0 (Y plane, interleaved UV)
    YVYU = 3,  // Packed YUV 4:2:2
    YUY2 = 4,  // Packed YUV 4:2:2
    UYVY = 5,  // Packed YUV 4:2:2
    RGBA = 6,  // Packed RGB
    BGRA = 7,  // Packed BGR
    BGRX = 8,  // Packed BGR (no alpha)
    Y800 = 9,  // Grayscale
    I444 = 10, // Planar YUV 4:4:4
    BGR3 = 11, // Packed BGR 24-bit
    I422 = 12, // Planar YUV 4:2:2
    I40A = 13, // Planar YUVA 4:2:0
    I42A = 14, // Planar YUVA 4:2:2
    YUVA = 15, // Planar YUVA 4:4:4
    AYUV = 16, // Packed YUVA 4:4:4
    I010 = 17, // Planar YUV 4:2:0 10-bit
    P010 = 18, // Semi-planar YUV 4:2:0 10-bit
    I210 = 19, // Planar YUV 4:2:2 10-bit
    I412 = 20, // Planar YUV 4:4:4 12-bit
    YA2L = 21, // Packed YUVA 4:2:2 10-bit
    P216 = 22, // Semi-planar YUV 4:2:2 16-bit
    P416 = 23, // Semi-planar YUV 4:4:4 16-bit
    R10L = 24, // Packed RGB 10-bit
}

impl VideoFormat {
    /// Returns number of planes for this format
    pub fn plane_count(self) -> usize {
        match self {
            VideoFormat::None => 0,
            VideoFormat::I420 | VideoFormat::I444 | VideoFormat::I422 => 3,
            VideoFormat::I40A | VideoFormat::I42A | VideoFormat::YUVA => 4,
            VideoFormat::NV12 | VideoFormat::P010 => 2,
            _ => 1,
        }
    }

    /// Returns bytes per pixel for packed formats
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            VideoFormat::RGBA | VideoFormat::BGRA | VideoFormat::BGRX | VideoFormat::AYUV => 4,
            VideoFormat::BGR3 => 3,
            VideoFormat::YVYU | VideoFormat::YUY2 | VideoFormat::UYVY => 2,
            VideoFormat::Y800 => 1,
            _ => 0, // Planar formats
        }
    }

    /// Check if format is planar
    pub fn is_planar(self) -> bool {
        self.plane_count() > 1
    }

    /// Calculate frame size in bytes
    pub fn calculate_size(self, width: u32, height: u32) -> usize {
        match self {
            VideoFormat::I420 | VideoFormat::NV12 => {
                let y_size = (width * height) as usize;
                let uv_size = ((width / 2) * (height / 2)) as usize;
                y_size + uv_size * 2
            }
            VideoFormat::I444 => (width * height * 3) as usize,
            VideoFormat::RGBA | VideoFormat::BGRA => (width * height * 4) as usize,
            _ => (width * height * self.bytes_per_pixel() as u32) as usize,
        }
    }
}

/// Video frame data structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub data: [*mut u8; 4],
    pub linesize: [u32; 4],
    pub width: u32,
    pub height: u32,
    pub format: VideoFormat,
    pub timestamp: u64,
}

unsafe impl Send for VideoFrame {}
unsafe impl Sync for VideoFrame {}

impl VideoFrame {
    pub fn new(width: u32, height: u32, format: VideoFormat) -> Self {
        Self {
            data: [std::ptr::null_mut(); 4],
            linesize: [0; 4],
            width,
            height,
            format,
            timestamp: 0,
        }
    }
}

/// Video output info
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VideoOutputInfo {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub format: u32, // VideoFormat as u32
}

/// Colorspace enumeration
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Default = 0,
    CS601 = 1,     // BT.601 (SD)
    CS709 = 2,     // BT.709 (HD)
    SRGB = 3,      // sRGB
    CS2100PQ = 4,  // BT.2100 PQ (HDR)
    CS2100HLG = 5, // BT.2100 HLG (HDR)
}

/// Color range enumeration
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRange {
    Default = 0,
    Partial = 1, // Limited range (16-235)
    Full = 2,    // Full range (0-255)
}
