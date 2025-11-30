//! Core type definitions for OBS scene compositor
//!
//! This module defines the data structures for scene items and their properties,
//! designed to be C-compatible for FFI while optimized for Rust performance.

use glam::{Mat4, Vec2};

/// Crop settings for a scene item
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SceneItemCrop {
    pub left: i32,
    pub right: i32,
    pub top: i32,
    pub bottom: i32,
}

impl SceneItemCrop {
    /// Check if any cropping is enabled
    pub fn is_enabled(&self) -> bool {
        self.left != 0 || self.right != 0 || self.top != 0 || self.bottom != 0
    }

    /// Calculate the cropped width given the original width
    pub fn calc_width(&self, original_width: u32) -> u32 {
        let crop_total = (self.left + self.right) as u32;
        if crop_total >= original_width {
            2 // Minimum size
        } else {
            original_width - crop_total
        }
    }

    /// Calculate the cropped height given the original height
    pub fn calc_height(&self, original_height: u32) -> u32 {
        let crop_total = (self.top + self.bottom) as u32;
        if crop_total >= original_height {
            2 // Minimum size
        } else {
            original_height - crop_total
        }
    }
}

/// Bounds type for scene items
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundsType {
    #[default]
    None = 0,
    Stretch = 1,
    ScaleInner = 2,
    ScaleOuter = 3,
    ScaleToWidth = 4,
    ScaleToHeight = 5,
    MaxOnly = 6,
}

/// Alignment flags (bitfield)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alignment(pub u32);

impl Alignment {
    pub const CENTER: u32 = 0;
    pub const LEFT: u32 = 1 << 0;
    pub const RIGHT: u32 = 1 << 1;
    pub const TOP: u32 = 1 << 2;
    pub const BOTTOM: u32 = 1 << 3;

    pub fn new(flags: u32) -> Self {
        Self(flags)
    }

    pub fn has_left(&self) -> bool {
        self.0 & Self::LEFT != 0
    }

    pub fn has_right(&self) -> bool {
        self.0 & Self::RIGHT != 0
    }

    pub fn has_top(&self) -> bool {
        self.0 & Self::TOP != 0
    }

    pub fn has_bottom(&self) -> bool {
        self.0 & Self::BOTTOM != 0
    }
}

impl Default for Alignment {
    fn default() -> Self {
        Self(Self::CENTER)
    }
}

/// Blend mode for scene items
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal = 0,
    Additive = 1,
    Subtract = 2,
    Screen = 3,
    Multiply = 4,
    Lighten = 5,
    Darken = 6,
}

/// Scale filter type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleFilter {
    #[default]
    Disable = 0,
    Point = 1,
    Bilinear = 2,
    Bicubic = 3,
    Lanczos = 4,
    Area = 5,
}

/// A single scene item with all its properties
#[derive(Debug, Clone)]
pub struct SceneItem {
    /// Unique identifier for this item
    pub id: i64,

    /// Reference to the source (opaque pointer for now, will be proper type in FFI)
    pub source_id: u64,

    // Transform properties
    /// Position in scene coordinates
    pub pos: Vec2,
    /// Scale factors (x, y)
    pub scale: Vec2,
    /// Rotation in degrees
    pub rotation: f32,
    /// Alignment flags
    pub alignment: Alignment,

    // Bounds and cropping
    /// Bounds type
    pub bounds_type: BoundsType,
    /// Bounds alignment
    pub bounds_align: Alignment,
    /// Bounds size
    pub bounds: Vec2,
    /// Crop to bounds
    pub crop_to_bounds: bool,
    /// Crop settings
    pub crop: SceneItemCrop,
    /// Bounds crop (calculated)
    pub bounds_crop: SceneItemCrop,

    // Cached transforms (dirty flag system)
    /// Cached draw transform matrix
    pub draw_transform: Mat4,
    /// Cached box transform matrix
    pub box_transform: Mat4,
    /// Output scale (cached)
    pub output_scale: Vec2,
    /// Box scale (cached)
    pub box_scale: Vec2,
    /// Transform needs recalculation
    pub transform_dirty: bool,

    // Last known source dimensions (for detecting changes)
    pub last_width: u32,
    pub last_height: u32,

    // Rendering properties
    /// Blend mode
    pub blend_mode: BlendMode,
    /// Scale filter
    pub scale_filter: ScaleFilter,

    // State flags
    /// Item is visible
    pub visible: bool,
    /// Item is locked (can't be moved/edited)
    pub locked: bool,
    /// Item is selected
    pub selected: bool,
    /// Item is a group
    pub is_group: bool,
}

impl SceneItem {
    /// Create a new scene item with default properties
    pub fn new(id: i64, source_id: u64) -> Self {
        Self {
            id,
            source_id,
            pos: Vec2::ZERO,
            scale: Vec2::ONE,
            rotation: 0.0,
            alignment: Alignment::default(),
            bounds_type: BoundsType::None,
            bounds_align: Alignment::default(),
            bounds: Vec2::ZERO,
            crop_to_bounds: false,
            crop: SceneItemCrop::default(),
            bounds_crop: SceneItemCrop::default(),
            draw_transform: Mat4::IDENTITY,
            box_transform: Mat4::IDENTITY,
            output_scale: Vec2::ONE,
            box_scale: Vec2::ZERO,
            transform_dirty: true,
            last_width: 0,
            last_height: 0,
            blend_mode: BlendMode::Normal,
            scale_filter: ScaleFilter::Disable,
            visible: true,
            locked: false,
            selected: false,
            is_group: false,
        }
    }

    /// Mark transform as dirty (needs recalculation)
    pub fn mark_transform_dirty(&mut self) {
        self.transform_dirty = true;
    }

    /// Check if source size has changed
    pub fn source_size_changed(&self, width: u32, height: u32) -> bool {
        self.last_width != width || self.last_height != height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_calculations() {
        let crop = SceneItemCrop {
            left: 10,
            right: 20,
            top: 5,
            bottom: 15,
        };

        assert!(crop.is_enabled());
        assert_eq!(crop.calc_width(100), 70); // 100 - 10 - 20
        assert_eq!(crop.calc_height(100), 80); // 100 - 5 - 15
    }

    #[test]
    fn test_crop_minimum_size() {
        let crop = SceneItemCrop {
            left: 50,
            right: 60,
            top: 0,
            bottom: 0,
        };

        assert_eq!(crop.calc_width(100), 2); // Exceeds original, return minimum
        assert_eq!(crop.calc_height(100), 100); // No vertical crop
    }

    #[test]
    fn test_alignment_flags() {
        let align = Alignment::new(Alignment::LEFT | Alignment::TOP);
        assert!(align.has_left());
        assert!(align.has_top());
        assert!(!align.has_right());
        assert!(!align.has_bottom());
    }

    #[test]
    fn test_scene_item_creation() {
        let item = SceneItem::new(1, 42);
        assert_eq!(item.id, 1);
        assert_eq!(item.source_id, 42);
        assert!(item.visible);
        assert!(!item.locked);
        assert!(item.transform_dirty);
        assert_eq!(item.scale, Vec2::ONE);
    }

    #[test]
    fn test_transform_dirty_flag() {
        let mut item = SceneItem::new(1, 42);
        item.transform_dirty = false;

        item.mark_transform_dirty();
        assert!(item.transform_dirty);
    }

    #[test]
    fn test_source_size_change_detection() {
        let mut item = SceneItem::new(1, 42);
        item.last_width = 1920;
        item.last_height = 1080;

        assert!(!item.source_size_changed(1920, 1080));
        assert!(item.source_size_changed(1280, 720));
    }
}
