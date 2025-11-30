# Hardware profile: i7-9700K (Coffee Lake, 9th gen Intel)
# Target: PC build (Windows)
# Features: AVX2, RTX 3050, 16GB RAM

$env:RUSTFLAGS="-C target-cpu=skylake -C target-feature=+avx2,+fma"
$env:CARGO_BUILD_TARGET="x86_64-pc-windows-msvc"

# Optimization flags for Coffee Lake
$env:OPT_LEVEL=3
$env:LTO="fat"
$env:CODEGEN_UNITS=1

# Memory configuration (16GB RAM)
$env:FRAME_POOL_SIZE=24
$env:TEXTURE_CACHE_SIZE=3072

Write-Host "Building for i7-9700K (Coffee Lake) - Windows"
Write-Host "Target CPU: skylake (Coffee Lake compatible)"
Write-Host "Features: AVX2, FMA"
Write-Host "GPU: RTX 3050"
Write-Host "RAM: 16GB"

Set-Location C:\path\to\OBSRust\obs-studio\rust-core
cargo build --release --workspace

Write-Host "Build complete: target\release\"
