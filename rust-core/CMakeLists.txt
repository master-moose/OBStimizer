# CMake integration for OBS Rust optimizations
# Integrates Rust libraries with OBS Studio build system

cmake_minimum_required(VERSION 3.16...3.25)

project(obs-rust-core)

# Options
option(ENABLE_RUST_OPTIMIZATIONS "Enable Rust-based optimizations" ON)
option(RUST_BUILD_RELEASE "Build Rust libraries in release mode" ON)
set(RUST_TARGET_CPU "native" CACHE STRING "Target CPU for Rust optimizations")

if(ENABLE_RUST_OPTIMIZATIONS)
    message(STATUS "Rust optimizations: ENABLED")
    message(STATUS "Rust target CPU: ${RUST_TARGET_CPU}")
    
    # Find Cargo
    find_program(CARGO cargo REQUIRED)
    if(NOT CARGO)
        message(FATAL_ERROR "Cargo not found. Please install Rust toolchain.")
    endif()
    
    # Find cbindgen for header generation
    find_program(CBINDGEN cbindgen)
    if(NOT CBINDGEN)
        message(WARNING "cbindgen not found. C headers will not be regenerated.")
    endif()
    
    # Set Rust build flags based on target CPU
    set(RUSTFLAGS "-C target-cpu=${RUST_TARGET_CPU}")
    
    # Additional flags for specific architectures
    if(RUST_TARGET_CPU STREQUAL "tigerlake")
        set(RUSTFLAGS "${RUSTFLAGS} -C target-feature=+avx2,+avx512f,+avx512dq")
    elseif(RUST_TARGET_CPU STREQUAL "skylake")
        set(RUSTFLAGS "${RUSTFLAGS} -C target-feature=+avx2,+fma")
    endif()
    
    # Build type
    if(RUST_BUILD_RELEASE)
        set(RUST_BUILD_TYPE "release")
        set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
    else()
        set(RUST_BUILD_TYPE "debug")
        set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/debug")
    endif()
    
    # Custom target to build Rust libraries
    add_custom_target(
        obs-rust-build
        ALL
        COMMAND ${CMAKE_COMMAND} -E env
            RUSTFLAGS=${RUSTFLAGS}
            ${CARGO} build --workspace --${RUST_BUILD_TYPE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust libraries in ${RUST_BUILD_TYPE} mode for ${RUST_TARGET_CPU}"
        BYPRODUCTS
            ${RUST_TARGET_DIR}/libobs_video.a
            ${RUST_TARGET_DIR}/libobs_compositor.a
            ${RUST_TARGET_DIR}/libobs_audio_mix.a
            ${RUST_TARGET_DIR}/libobs_nvenc.a
            ${RUST_TARGET_DIR}/libobs_ffi.a
    )
    
    # Generate C headers if cbindgen is available
    if(CBINDGEN)
        add_custom_command(
            TARGET obs-rust-build
            POST_BUILD
            COMMAND ${CBINDGEN}
                --config ${CMAKE_CURRENT_SOURCE_DIR}/obs-ffi/cbindgen.toml
                --crate obs-ffi
                --output ${CMAKE_CURRENT_SOURCE_DIR}/obs-ffi/obs-rust.h
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/obs-ffi
            COMMENT "Generating C headers with cbindgen"
        )
    endif()
    
    # Create imported libraries
    add_library(obs-rust-video STATIC IMPORTED GLOBAL)
    set_target_properties(obs-rust-video PROPERTIES
        IMPORTED_LOCATION ${RUST_TARGET_DIR}/libobs_video.a
    )
    
    add_library(obs-rust-compositor STATIC IMPORTED GLOBAL)
    set_target_properties(obs-rust-compositor PROPERTIES
        IMPORTED_LOCATION ${RUST_TARGET_DIR}/libobs_compositor.a
    )
    
    add_library(obs-rust-audio-mix STATIC IMPORTED GLOBAL)
    set_target_properties(obs-rust-audio-mix PROPERTIES
        IMPORTED_LOCATION ${RUST_TARGET_DIR}/libobs_audio_mix.a
    )
    
    add_library(obs-rust-nvenc STATIC IMPORTED GLOBAL)
    set_target_properties(obs-rust-nvenc PROPERTIES
        IMPORTED_LOCATION ${RUST_TARGET_DIR}/libobs_nvenc.a
    )
    
    add_library(obs-rust-ffi STATIC IMPORTED GLOBAL)
    set_target_properties(obs-rust-ffi PROPERTIES
        IMPORTED_LOCATION ${RUST_TARGET_DIR}/libobs_ffi.a
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/obs-ffi
    )
    
    # Ensure Rust libraries are built before being used
    add_dependencies(obs-rust-video obs-rust-build)
    add_dependencies(obs-rust-compositor obs-rust-build)
    add_dependencies(obs-rust-audio-mix obs-rust-build)
    add_dependencies(obs-rust-nvenc obs-rust-build)
    add_dependencies(obs-rust-ffi obs-rust-build)
    
    # Create interface library that links all Rust components
    add_library(obs-rust INTERFACE)
    target_link_libraries(obs-rust INTERFACE
        obs-rust-ffi
        obs-rust-video
        obs-rust-compositor
        obs-rust-audio-mix
        obs-rust-nvenc
    )
    
    # Add Rust library dependencies (pthread, dl on Linux)
    if(UNIX AND NOT APPLE)
        target_link_libraries(obs-rust INTERFACE pthread dl m)
    elseif(WIN32)
        target_link_libraries(obs-rust INTERFACE userenv ws2_32 bcrypt)
    endif()
    
    message(STATUS "Rust libraries will be built and linked into OBS")
    
else()
    message(STATUS "Rust optimizations: DISABLED")
endif()
