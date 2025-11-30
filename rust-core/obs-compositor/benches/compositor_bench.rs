//! Benchmarks for obs-compositor
//!
//! Measures performance of scene operations and transform calculations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::Vec2;
use obs_compositor::{render_scene, Scene, SceneItem};

fn bench_scene_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("scene_iteration");

    for item_count in [10, 50, 100, 200].iter() {
        let scene = Scene::new(1920, 1080);

        // Populate scene
        for i in 0..*item_count {
            scene.add_item(SceneItem::new(0, i as u64));
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(item_count),
            item_count,
            |b, _| {
                b.iter(|| {
                    let mut count = 0;
                    scene.render_items(|_item| {
                        count += 1;
                    });
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

fn bench_transform_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_calculation");

    // Benchmark with dirty flag enabled (recalculates)
    group.bench_function("with_recalc", |b| {
        let scene = Scene::new(1920, 1080);
        let id = scene.add_item(SceneItem::new(0, 100));

        b.iter(|| {
            // Mark dirty to force recalculation
            scene.update_item(id, |item| {
                item.mark_transform_dirty();
            });

            scene.update_transforms(&[(100, 1920, 1080)]);
        });
    });

    // Benchmark with dirty flag disabled (cached)
    group.bench_function("cached", |b| {
        let scene = Scene::new(1920, 1080);
        scene.add_item(SceneItem::new(0, 100));

        // Initial calculation
        scene.update_transforms(&[(100, 1920, 1080)]);

        b.iter(|| {
            // Should skip recalculation
            scene.update_transforms(&[(100, 1920, 1080)]);
        });
    });

    group.finish();
}

fn bench_render_scene(c: &mut Criterion) {
    let mut group = c.benchmark_group("render_scene");

    for item_count in [10, 50, 100].iter() {
        let scene = Scene::new(1920, 1080);

        for i in 0..*item_count {
            let mut item = SceneItem::new(0, i as u64);
            item.pos = Vec2::new((i * 10) as f32, (i * 10) as f32);
            item.scale = Vec2::new(1.5, 1.5);
            item.rotation = (i as f32) * 0.1;
            scene.add_item(item);
        }

        // Update transforms once
        let dimensions: Vec<_> = (0..*item_count).map(|i| (i as u64, 1920, 1080)).collect();
        scene.update_transforms(&dimensions);

        group.bench_with_input(
            BenchmarkId::from_parameter(item_count),
            item_count,
            |b, _| {
                b.iter(|| {
                    let commands = render_scene(&scene);
                    black_box(commands);
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_rendering(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let scene = Arc::new(Scene::new(1920, 1080));

    // Add items
    for i in 0..50 {
        scene.add_item(SceneItem::new(0, i));
    }

    c.bench_function("concurrent_4_threads", |b| {
        b.iter(|| {
            let mut handles = vec![];

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

            for handle in handles {
                black_box(handle.join().unwrap());
            }
        });
    });
}

fn bench_item_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("item_operations");

    group.bench_function("add_item", |b| {
        let scene = Scene::new(1920, 1080);
        let mut counter = 0u64;

        b.iter(|| {
            let item = SceneItem::new(0, counter);
            scene.add_item(item);
            counter += 1;
        });
    });

    group.bench_function("remove_item", |b| {
        let scene = Scene::new(1920, 1080);

        // Pre-populate
        let ids: Vec<_> = (0..1000)
            .map(|i| scene.add_item(SceneItem::new(0, i)))
            .collect();

        let mut idx = 0;
        b.iter(|| {
            scene.remove_item(ids[idx % ids.len()]);
            idx += 1;
        });
    });

    group.bench_function("update_item", |b| {
        let scene = Scene::new(1920, 1080);
        let id = scene.add_item(SceneItem::new(0, 100));

        b.iter(|| {
            scene.update_item(id, |item| {
                item.pos.x += 1.0;
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scene_iteration,
    bench_transform_calculation,
    bench_render_scene,
    bench_concurrent_rendering,
    bench_item_operations
);
criterion_main!(benches);
