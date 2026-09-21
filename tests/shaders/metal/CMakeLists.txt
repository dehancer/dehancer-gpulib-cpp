set(METAL_KERNELS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/TestKernels_metal.metallib")
configure_file(paths_config.h.in "${PROJECT_BINARY_DIR}/generated/tests/metal/paths_config.h" @ONLY)
dehancer_add_metal_shaders(TestKernels_metal "${CMAKE_CURRENT_SOURCE_DIR}/TestKernels.metal")
