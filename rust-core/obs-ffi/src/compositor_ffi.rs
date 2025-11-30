//! Scene compositor FFI bindings
//!
//! C-compatible API for the Rust scene compositor.

use obs_compositor::{Scene, SceneItem};
use std::os::raw::{c_int, c_void};

/// Opaque handle to Scene (C-compatible)
pub struct OBSScene {
    _private: [u8; 0],
}

/// C-compatible scene item structure
#[repr(C)]
pub struct CSceneItem {
    pub id: i64,
    pub source_id: u64,
    pub pos_x: f32,
    pub pos_y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotation: f32,
    pub visible: bool,
    pub locked: bool,
}

// ============================================================================
// SCENE COMPOSITOR API
// ============================================================================

/// Create a new scene
///
/// # Safety
/// Safe to call.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_scene_create(width: u32, height: u32) -> *mut OBSScene {
    let scene = Box::new(Scene::new(width, height));
    Box::into_raw(scene) as *mut OBSScene
}

/// Destroy a scene
///
/// # Safety
/// Caller must ensure ptr is valid and not already freed.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_scene_destroy(ptr: *mut OBSScene) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut Scene);
    }
}

/// Add an item to the scene
///
/// # Safety
/// Caller must ensure ptr and item are valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_scene_add_item(
    ptr: *mut OBSScene,
    item: *const CSceneItem,
) -> i64 {
    if ptr.is_null() || item.is_null() {
        return -1;
    }

    let scene = &*(ptr as *const Scene);

    let mut scene_item = SceneItem::new(0, (*item).source_id);
    scene_item.pos = glam::Vec2::new((*item).pos_x, (*item).pos_y);
    scene_item.scale = glam::Vec2::new((*item).scale_x, (*item).scale_y);
    scene_item.rotation = (*item).rotation;
    scene_item.visible = (*item).visible;
    scene_item.locked = (*item).locked;

    scene.add_item(scene_item)
}

/// Remove an item from the scene
///
/// # Safety
/// Caller must ensure ptr is valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_scene_remove_item(ptr: *mut OBSScene, id: i64) -> c_int {
    if ptr.is_null() {
        return 0;
    }

    let scene = &*(ptr as *const Scene);
    if scene.remove_item(id) {
        1
    } else {
        0
    }
}

/// Get the number of items in the scene
///
/// # Safety
/// Caller must ensure ptr is valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_scene_item_count(ptr: *const OBSScene) -> usize {
    if ptr.is_null() {
        return 0;
    }

    let scene = &*(ptr as *const Scene);
    scene.item_count()
}

/// Update transforms for all scene items
///
/// # Safety
/// Caller must ensure pointers are valid and dimensions array is properly sized.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_scene_update_transforms(
    ptr: *mut OBSScene,
    dimensions: *const (u64, u32, u32),
    count: usize,
) {
    if ptr.is_null() || dimensions.is_null() {
        return;
    }

    let scene = &*(ptr as *const Scene);
    let dims_slice = std::slice::from_raw_parts(dimensions, count);
    scene.update_transforms(dims_slice);
}

/// Render all visible items in the scene
///
/// # Safety
/// Caller must ensure ptr and callback are valid.
#[no_mangle]
pub unsafe extern "C" fn obs_rust_scene_render(
    ptr: *const OBSScene,
    callback: extern "C" fn(u64, *const f32, u32),
    _user_data: *mut c_void,
) {
    if ptr.is_null() {
        return;
    }

    let scene = &*(ptr as *const Scene);
    scene.render_items(|item| {
        // Pass transform matrix as flat array
        let matrix = item.draw_transform.to_cols_array();
        callback(item.source_id, matrix.as_ptr(), item.blend_mode as u32);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_ffi() {
        unsafe {
            let scene = obs_rust_scene_create(1920, 1080);
            assert!(!scene.is_null());

            let item = CSceneItem {
                id: 0,
                source_id: 100,
                pos_x: 0.0,
                pos_y: 0.0,
                scale_x: 1.0,
                scale_y: 1.0,
                rotation: 0.0,
                visible: true,
                locked: false,
            };

            let id = obs_rust_scene_add_item(scene, &item);
            assert!(id > 0);

            let count = obs_rust_scene_item_count(scene);
            assert_eq!(count, 1);

            let removed = obs_rust_scene_remove_item(scene, id);
            assert_eq!(removed, 1);

            obs_rust_scene_destroy(scene);
        }
    }
}
