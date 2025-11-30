//! Scene management with array-based storage and lock-free rendering
//!
//! This module provides the core Scene type that manages scene items using
//! Vec-based storage for cache-friendly iteration and RwLock for concurrent reads.

use crate::transform::update_item_transform;
use crate::types::SceneItem;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// A scene that contains multiple scene items
///
/// Uses array-based storage (Vec) instead of linked lists for better cache locality.
/// RwLock allows multiple concurrent readers (rendering threads) while ensuring
/// safe writes (add/remove/reorder operations).
pub struct Scene {
    /// Array-based storage for scene items (replaces linked list)
    items: Arc<RwLock<Vec<SceneItem>>>,

    /// Scene dimensions
    width: u32,
    height: u32,

    /// Whether this is a group
    is_group: bool,

    /// ID counter for generating unique item IDs
    id_counter: AtomicI64,

    /// Statistics
    render_count: AtomicU64,
}

impl Scene {
    /// Create a new scene with the given dimensions
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            items: Arc::new(RwLock::new(Vec::with_capacity(16))), // Pre-allocate for typical scenes
            width,
            height,
            is_group: false,
            id_counter: AtomicI64::new(1),
            render_count: AtomicU64::new(0),
        }
    }

    /// Create a new group (special scene type)
    pub fn new_group() -> Self {
        let mut scene = Self::new(0, 0);
        scene.is_group = true;
        scene
    }

    /// Get scene dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Set scene dimensions
    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Check if this is a group
    pub fn is_group(&self) -> bool {
        self.is_group
    }

    /// Add a new item to the scene
    ///
    /// Returns the ID of the newly added item.
    pub fn add_item(&self, mut item: SceneItem) -> i64 {
        let id = self.id_counter.fetch_add(1, Ordering::SeqCst);
        item.id = id;
        item.mark_transform_dirty();

        let mut items = self.items.write();
        items.push(item);
        id
    }

    /// Remove an item by ID
    ///
    /// Returns true if the item was found and removed.
    pub fn remove_item(&self, id: i64) -> bool {
        let mut items = self.items.write();
        if let Some(pos) = items.iter().position(|item| item.id == id) {
            items.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get the number of items in the scene
    pub fn item_count(&self) -> usize {
        self.items.read().len()
    }

    /// Reorder items based on a list of (id, new_index) pairs
    ///
    /// This is more efficient than individual moves for bulk reordering.
    pub fn reorder_items(&self, order: &[(i64, usize)]) {
        let mut items = self.items.write();
        let mut new_items = Vec::with_capacity(items.len());

        // Create a map of id -> item
        let mut item_map: std::collections::HashMap<i64, SceneItem> =
            items.drain(..).map(|item| (item.id, item)).collect();

        // Build new order
        for (id, _) in order {
            if let Some(item) = item_map.remove(id) {
                new_items.push(item);
            }
        }

        // Add any remaining items that weren't in the order list
        new_items.extend(item_map.into_values());

        *items = new_items;
    }

    /// Move an item to a new position
    pub fn move_item(&self, id: i64, new_index: usize) -> bool {
        let mut items = self.items.write();
        if let Some(old_index) = items.iter().position(|item| item.id == id) {
            if new_index >= items.len() {
                return false;
            }
            let item = items.remove(old_index);
            items.insert(new_index, item);
            true
        } else {
            false
        }
    }

    /// Update an item's properties
    ///
    /// The callback receives a mutable reference to the item if found.
    pub fn update_item<F>(&self, id: i64, mut update_fn: F) -> bool
    where
        F: FnMut(&mut SceneItem),
    {
        let mut items = self.items.write();
        if let Some(item) = items.iter_mut().find(|item| item.id == id) {
            update_fn(item);
            item.mark_transform_dirty();
            true
        } else {
            false
        }
    }

    /// Render all visible items (lock-free read)
    ///
    /// This function acquires a read lock, allowing multiple threads to render
    /// concurrently. The callback is invoked for each visible item.
    pub fn render_items<F>(&self, mut callback: F)
    where
        F: FnMut(&SceneItem),
    {
        self.render_count.fetch_add(1, Ordering::Relaxed);

        let items = self.items.read();
        for item in items.iter() {
            if item.visible {
                callback(item);
            }
        }
    }

    /// Update transforms for all items that need it
    ///
    /// This should be called before rendering to ensure transforms are up-to-date.
    /// Only items with dirty transforms are recalculated.
    pub fn update_transforms(&self, source_dimensions: &[(u64, u32, u32)]) {
        let mut items = self.items.write();

        // Create a map of source_id -> (width, height) for quick lookup
        let dim_map: std::collections::HashMap<u64, (u32, u32)> = source_dimensions
            .iter()
            .map(|(id, w, h)| (*id, (*w, *h)))
            .collect();

        for item in items.iter_mut() {
            if let Some(&(width, height)) = dim_map.get(&item.source_id) {
                // Update transform if dirty or source size changed
                if item.transform_dirty || item.source_size_changed(width, height) {
                    update_item_transform(item, width, height);
                }
            }
        }
    }

    /// Get statistics
    pub fn render_count(&self) -> u64 {
        self.render_count.load(Ordering::Relaxed)
    }

    /// Get a snapshot of all items (for debugging/inspection)
    pub fn get_items_snapshot(&self) -> Vec<SceneItem> {
        self.items.read().clone()
    }

    /// Find an item by ID
    pub fn find_item(&self, id: i64) -> Option<SceneItem> {
        self.items.read().iter().find(|item| item.id == id).cloned()
    }
}

// Implement Clone for Scene (creates a new Arc reference to the same data)
impl Clone for Scene {
    fn clone(&self) -> Self {
        Self {
            items: Arc::clone(&self.items),
            width: self.width,
            height: self.height,
            is_group: self.is_group,
            id_counter: AtomicI64::new(self.id_counter.load(Ordering::SeqCst)),
            render_count: AtomicU64::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_creation() {
        let scene = Scene::new(1920, 1080);
        assert_eq!(scene.dimensions(), (1920, 1080));
        assert!(!scene.is_group());
        assert_eq!(scene.item_count(), 0);
    }

    #[test]
    fn test_group_creation() {
        let scene = Scene::new_group();
        assert!(scene.is_group());
    }

    #[test]
    fn test_add_remove_items() {
        let scene = Scene::new(1920, 1080);

        let item1 = SceneItem::new(0, 100);
        let item2 = SceneItem::new(0, 200);

        let id1 = scene.add_item(item1);
        let id2 = scene.add_item(item2);

        assert_eq!(scene.item_count(), 2);

        assert!(scene.remove_item(id1));
        assert_eq!(scene.item_count(), 1);

        assert!(!scene.remove_item(id1)); // Already removed
        assert!(scene.remove_item(id2));
        assert_eq!(scene.item_count(), 0);
    }

    #[test]
    fn test_move_item() {
        let scene = Scene::new(1920, 1080);

        let id1 = scene.add_item(SceneItem::new(0, 100));
        let id2 = scene.add_item(SceneItem::new(0, 200));
        let id3 = scene.add_item(SceneItem::new(0, 300));

        // Move item 1 to position 2
        assert!(scene.move_item(id1, 2));

        let items = scene.get_items_snapshot();
        assert_eq!(items[0].id, id2);
        assert_eq!(items[1].id, id3);
        assert_eq!(items[2].id, id1);
    }

    #[test]
    fn test_update_item() {
        let scene = Scene::new(1920, 1080);
        let id = scene.add_item(SceneItem::new(0, 100));

        assert!(scene.update_item(id, |item| {
            item.visible = false;
            item.locked = true;
        }));

        let item = scene.find_item(id).unwrap();
        assert!(!item.visible);
        assert!(item.locked);
    }

    #[test]
    fn test_render_items() {
        let scene = Scene::new(1920, 1080);

        let id1 = scene.add_item(SceneItem::new(0, 100));
        let id2 = scene.add_item(SceneItem::new(0, 200));

        // Hide one item
        scene.update_item(id2, |item| item.visible = false);

        let mut rendered_ids = Vec::new();
        scene.render_items(|item| {
            rendered_ids.push(item.id);
        });

        assert_eq!(rendered_ids.len(), 1);
        assert_eq!(rendered_ids[0], id1);
    }

    #[test]
    fn test_concurrent_rendering() {
        use std::thread;

        let scene = Arc::new(Scene::new(1920, 1080));
        scene.add_item(SceneItem::new(0, 100));
        scene.add_item(SceneItem::new(0, 200));

        let mut handles = vec![];

        // Spawn multiple reader threads
        for _ in 0..4 {
            let scene_clone = Arc::clone(&scene);
            handles.push(thread::spawn(move || {
                let mut count = 0;
                scene_clone.render_items(|_| {
                    count += 1;
                });
                count
            }));
        }

        // All threads should see 2 items
        for handle in handles {
            assert_eq!(handle.join().unwrap(), 2);
        }
    }

    #[test]
    fn test_update_transforms() {
        let scene = Scene::new(1920, 1080);
        let id = scene.add_item(SceneItem::new(0, 100));

        // Provide source dimensions
        let dimensions = vec![(100, 1920, 1080)];
        scene.update_transforms(&dimensions);

        let item = scene.find_item(id).unwrap();
        assert!(!item.transform_dirty);
        assert_eq!(item.last_width, 1920);
        assert_eq!(item.last_height, 1080);
    }
}
