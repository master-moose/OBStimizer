# Hardware profile: i7-1165G7 (Tiger Lake, 11th gen Intel)
# Target: Laptop build (Linux)
# Features: AVX2, AVX-512, 24GB RAM

export RUSTFLAGS="-C target-cpu=tigerlake -C target-feature=+avx2,+avx512f,+avx512dq"
export CARGO_BUILD_TARGET="x86_64-unknown-linux-gnu"

# Optimization flags for Tiger Lake
export OPT_LEVEL=3
export LTO="fat"
export CODEGEN_UNITS=1

# Memory configuration (24GB RAM)
export FRAME_POOL_SIZE=32
export TEXTURE_CACHE_SIZE=4096

echo "Building for i7-1165G7 (Tiger Lake) - Linux"
echo "Target CPU: tigerlake"
echo "Features: AVX2, AVX-512"
echo "RAM: 24GB"

cd /home/jai/Projects/OBSRust/obs-studio/rust-core
cargo build --release --workspace

echo "Build complete: target/release/"
