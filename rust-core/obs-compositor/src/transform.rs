//! Transform calculation and caching for scene items
//!
//! This module provides SIMD-optimized transform calculations using glam,
//! with a dirty flag system to prevent redundant recalculations.

use crate::types::{Alignment, BoundsType, SceneItem, SceneItemCrop};
use glam::{Mat4, Vec2};

/// Add alignment offset to a position
fn add_alignment(pos: &mut Vec2, align: Alignment, cx: i32, cy: i32) {
    if align.has_right() {
        pos.x += cx as f32;
    } else if !align.has_left() {
        pos.x += (cx / 2) as f32;
    }

    if align.has_bottom() {
        pos.y += cy as f32;
    } else if !align.has_top() {
        pos.y += (cy / 2) as f32;
    }
}

/// Calculate bounds data for a scene item
fn calculate_bounds_data(
    item: &mut SceneItem,
    origin: &mut Vec2,
    scale: &mut Vec2,
    cx: &mut u32,
    cy: &mut u32,
) {
    let bounds = item.bounds;
    let width = (*cx as f32) * scale.x.abs();
    let height = (*cy as f32) * scale.y.abs();
    let item_aspect = width / height;
    let bounds_aspect = bounds.x / bounds.y;
    let mut bounds_type = item.bounds_type;

    // MaxOnly becomes ScaleInner if size exceeds bounds
    if bounds_type == BoundsType::MaxOnly && (width > bounds.x || height > bounds.y) {
        bounds_type = BoundsType::ScaleInner;
    }

    match bounds_type {
        BoundsType::ScaleInner | BoundsType::ScaleOuter => {
            let mut use_width = bounds_aspect < item_aspect;
            if bounds_type == BoundsType::ScaleOuter {
                use_width = !use_width;
            }

            let mul = if use_width {
                bounds.x / width
            } else {
                bounds.y / height
            };

            *scale *= mul;
        }
        BoundsType::ScaleToWidth => {
            *scale *= bounds.x / width;
        }
        BoundsType::ScaleToHeight => {
            *scale *= bounds.y / height;
        }
        BoundsType::Stretch => {
            scale.x = bounds.x / (*cx as f32) * scale.x.signum();
            scale.y = bounds.y / (*cy as f32) * scale.y.signum();
        }
        _ => {}
    }

    let new_width = (*cx as f32) * scale.x;
    let new_height = (*cy as f32) * scale.y;

    let width_diff = bounds.x - new_width.abs();
    let height_diff = bounds.y - new_height.abs();
    *cx = bounds.x as u32;
    *cy = bounds.y as u32;

    add_alignment(
        origin,
        item.bounds_align,
        -width_diff as i32,
        -height_diff as i32,
    );

    // Set cropping if enabled and large enough size difference exists
    if item.crop_to_bounds && (width_diff < -0.1 || height_diff < -0.1) {
        let crop_width = width_diff < -0.1;
        let crop_flipped = if crop_width {
            new_width < 0.0
        } else {
            new_height < 0.0
        };

        let crop_diff = if crop_width { width_diff } else { height_diff };
        let crop_scale = if crop_width { scale.x } else { scale.y };

        let crop_align_mask = if crop_width {
            Alignment::LEFT | Alignment::RIGHT
        } else {
            Alignment::TOP | Alignment::BOTTOM
        };
        let crop_align = Alignment::new(item.bounds_align.0 & crop_align_mask);

        let overdraw = (crop_diff / crop_scale).abs();

        let overdraw_tl = if crop_align.0 & (Alignment::TOP | Alignment::LEFT) != 0 {
            0.0
        } else if crop_align.0 & (Alignment::BOTTOM | Alignment::RIGHT) != 0 {
            overdraw
        } else {
            overdraw / 2.0
        };

        let overdraw_br = overdraw - overdraw_tl;

        let (crop_br, crop_tl) = if crop_flipped {
            (overdraw_tl.round() as i32, overdraw_br.round() as i32)
        } else {
            (overdraw_br.round() as i32, overdraw_tl.round() as i32)
        };

        if crop_width {
            item.bounds_crop.right = crop_br;
            item.bounds_crop.left = crop_tl;
        } else {
            item.bounds_crop.bottom = crop_br;
            item.bounds_crop.top = crop_tl;
        }
    }

    // Makes the item stay in-place in the box if flipped
    origin.x += if new_width < 0.0 { new_width } else { 0.0 };
    origin.y += if new_height < 0.0 { new_height } else { 0.0 };
}

/// Calculate the cropped width
fn calc_cx(item: &SceneItem, width: u32) -> u32 {
    let crop_cx =
        (item.crop.left + item.crop.right + item.bounds_crop.left + item.bounds_crop.right) as u32;
    if crop_cx > width {
        2
    } else {
        width - crop_cx
    }
}

/// Calculate the cropped height
fn calc_cy(item: &SceneItem, height: u32) -> u32 {
    let crop_cy =
        (item.crop.top + item.crop.bottom + item.bounds_crop.top + item.bounds_crop.bottom) as u32;
    if crop_cy > height {
        2
    } else {
        height - crop_cy
    }
}

/// Update the transform matrices for a scene item
///
/// This function recalculates the draw_transform and box_transform matrices
/// using SIMD-optimized operations from glam. It respects the dirty flag
/// to avoid redundant calculations.
///
/// # Arguments
/// * `item` - The scene item to update
/// * `source_width` - Current width of the source
/// * `source_height` - Current height of the source
pub fn update_item_transform(item: &mut SceneItem, source_width: u32, source_height: u32) {
    // Skip if transform hasn't changed
    if !item.transform_dirty {
        return;
    }

    // Reset bounds crop
    item.bounds_crop = SceneItemCrop::default();

    let width = source_width;
    let height = source_height;

    let mut cx = calc_cx(item, width);
    let mut cy = calc_cy(item, height);
    item.last_width = width;
    item.last_height = height;

    let width = cx;
    let height = cy;

    let mut base_origin = Vec2::ZERO;
    let mut origin = Vec2::ZERO;
    let mut scale = item.scale;
    let position = item.pos;

    // Calculate bounds if enabled
    if item.bounds_type != BoundsType::None {
        calculate_bounds_data(item, &mut origin, &mut scale, &mut cx, &mut cy);
    } else {
        cx = (width as f32 * scale.x) as u32;
        cy = (height as f32 * scale.y) as u32;
    }

    add_alignment(&mut origin, item.alignment, cx as i32, cy as i32);

    // Build draw transform using SIMD-optimized glam operations
    // Order: translate(position) * rotate(rotation) * translate(-origin) * scale(scale)
    item.draw_transform = Mat4::from_translation(position.extend(0.0))
        * Mat4::from_rotation_z(item.rotation.to_radians())
        * Mat4::from_translation((-origin).extend(0.0))
        * Mat4::from_scale(scale.extend(1.0));

    item.output_scale = scale;

    // Calculate box transform
    let box_scale = if item.bounds_type != BoundsType::None {
        item.bounds
    } else {
        Vec2::new(scale.x * width as f32, scale.y * height as f32)
    };

    item.box_scale = box_scale;

    add_alignment(
        &mut base_origin,
        item.alignment,
        box_scale.x as i32,
        box_scale.y as i32,
    );

    item.box_transform = Mat4::from_translation(position.extend(0.0))
        * Mat4::from_rotation_z(item.rotation.to_radians())
        * Mat4::from_translation((-base_origin).extend(0.0))
        * Mat4::from_scale(box_scale.extend(1.0));

    item.transform_dirty = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_alignment_center() {
        let mut pos = Vec2::ZERO;
        add_alignment(&mut pos, Alignment::new(Alignment::CENTER), 100, 50);
        assert_eq!(pos, Vec2::new(50.0, 25.0));
    }

    #[test]
    fn test_add_alignment_top_left() {
        let mut pos = Vec2::ZERO;
        add_alignment(
            &mut pos,
            Alignment::new(Alignment::LEFT | Alignment::TOP),
            100,
            50,
        );
        assert_eq!(pos, Vec2::ZERO);
    }

    #[test]
    fn test_add_alignment_bottom_right() {
        let mut pos = Vec2::ZERO;
        add_alignment(
            &mut pos,
            Alignment::new(Alignment::RIGHT | Alignment::BOTTOM),
            100,
            50,
        );
        assert_eq!(pos, Vec2::new(100.0, 50.0));
    }

    #[test]
    fn test_calc_cx_cy() {
        let mut item = SceneItem::new(1, 42);
        item.crop.left = 10;
        item.crop.right = 20;
        item.crop.top = 5;
        item.crop.bottom = 15;

        assert_eq!(calc_cx(&item, 100), 70); // 100 - 10 - 20
        assert_eq!(calc_cy(&item, 100), 80); // 100 - 5 - 15
    }

    #[test]
    fn test_update_transform_basic() {
        let mut item = SceneItem::new(1, 42);
        item.pos = Vec2::new(100.0, 200.0);
        item.scale = Vec2::new(2.0, 2.0);
        item.rotation = 0.0;

        update_item_transform(&mut item, 1920, 1080);

        assert!(!item.transform_dirty);
        assert_eq!(item.last_width, 1920);
        assert_eq!(item.last_height, 1080);
        assert_eq!(item.output_scale, Vec2::new(2.0, 2.0));
    }

    #[test]
    fn test_transform_dirty_flag_skip() {
        let mut item = SceneItem::new(1, 42);
        update_item_transform(&mut item, 1920, 1080);

        // Mark as clean
        item.transform_dirty = false;
        let old_transform = item.draw_transform;

        // Should skip recalculation
        update_item_transform(&mut item, 1920, 1080);
        assert_eq!(item.draw_transform, old_transform);
    }

    #[test]
    fn test_bounds_stretch() {
        let mut item = SceneItem::new(1, 42);
        item.bounds_type = BoundsType::Stretch;
        item.bounds = Vec2::new(640.0, 360.0);
        item.scale = Vec2::ONE;

        update_item_transform(&mut item, 1920, 1080);

        assert!(!item.transform_dirty);
        // Scale should be adjusted to match bounds
        assert!(item.output_scale.x > 0.0);
        assert!(item.output_scale.y > 0.0);
    }
}
