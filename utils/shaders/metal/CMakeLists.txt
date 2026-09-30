set(METAL_KERNELS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/UtilsKernels.metallib")
configure_file(paths_config.h.in "${PROJECT_BINARY_DIR}/generated/utils/metal/paths_config.h" @ONLY)
dehancer_add_metal_shaders(UtilsKernels "${CMAKE_CURRENT_SOURCE_DIR}/UtilsKernels.metal")
