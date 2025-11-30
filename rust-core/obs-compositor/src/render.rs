//! Rendering pipeline for scene compositor
//!
//! This module provides the rendering logic that converts scene items
//! into render commands for the graphics backend.

use crate::types::{BlendMode, SceneItem};
use glam::Mat4;

/// A render command for a single scene item
#[derive(Debug, Clone)]
pub struct RenderCommand {
    /// Source ID to render
    pub source_id: u64,

    /// Transform matrix to apply
    pub transform: Mat4,

    /// Blend mode for compositing
    pub blend_mode: BlendMode,

    /// Whether to use the item's texture render
    pub use_item_texture: bool,
}

impl RenderCommand {
    /// Create a new render command from a scene item
    pub fn from_item(item: &SceneItem) -> Self {
        Self {
            source_id: item.source_id,
            transform: item.draw_transform,
            blend_mode: item.blend_mode,
            use_item_texture: item.crop.is_enabled()
                || item.bounds_crop.is_enabled()
                || item.scale_filter != crate::types::ScaleFilter::Disable,
        }
    }
}

/// Render a scene and return a list of render commands
///
/// This function iterates through all visible items in the scene
/// and generates render commands in back-to-front order.
pub fn render_scene(scene: &crate::scene::Scene) -> Vec<RenderCommand> {
    let mut commands = Vec::new();

    scene.render_items(|item| {
        commands.push(RenderCommand::from_item(item));
    });

    commands
}

/// Render a scene with a custom filter predicate
///
/// Only items that pass the filter will be rendered.
pub fn render_scene_filtered<F>(scene: &crate::scene::Scene, mut filter: F) -> Vec<RenderCommand>
where
    F: FnMut(&SceneItem) -> bool,
{
    let mut commands = Vec::new();

    scene.render_items(|item| {
        if filter(item) {
            commands.push(RenderCommand::from_item(item));
        }
    });

    commands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;
    use crate::types::SceneItem;

    #[test]
    fn test_render_command_creation() {
        let item = SceneItem::new(1, 42);
        let cmd = RenderCommand::from_item(&item);

        assert_eq!(cmd.source_id, 42);
        assert_eq!(cmd.blend_mode, BlendMode::Normal);
    }

    #[test]
    fn test_render_scene() {
        let scene = Scene::new(1920, 1080);
        scene.add_item(SceneItem::new(0, 100));
        scene.add_item(SceneItem::new(0, 200));

        let commands = render_scene(&scene);
        assert_eq!(commands.len(), 2);
    }

    #[test]
    fn test_render_scene_filtered() {
        let scene = Scene::new(1920, 1080);
        let id1 = scene.add_item(SceneItem::new(0, 100));
        scene.add_item(SceneItem::new(0, 200));

        // Only render items with specific ID
        let commands = render_scene_filtered(&scene, |item| item.id == id1);
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].source_id, 100);
    }

    #[test]
    fn test_item_texture_detection() {
        let mut item = SceneItem::new(1, 42);

        // No crop, no special filter
        let cmd = RenderCommand::from_item(&item);
        assert!(!cmd.use_item_texture);

        // With crop enabled
        item.crop.left = 10;
        let cmd = RenderCommand::from_item(&item);
        assert!(cmd.use_item_texture);
    }
}
