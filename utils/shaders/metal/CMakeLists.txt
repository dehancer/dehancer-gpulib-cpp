set(METAL_KERNELS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/UtilsKernels_metal.metallib")
configure_file(paths_config.h.in "${PROJECT_BINARY_DIR}/generated/utils/metal/paths_config.h" @ONLY)
dehancer_add_metal_shaders(UtilsKernels_metal "${CMAKE_CURRENT_SOURCE_DIR}/UtilsKernels.metal")
