set(CUDA_KERNELS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/UtilsKernels_cuda.fatbin")
set(CUDA_IMAGE_DIR "${PROJECT_SOURCE_DIR}/tests/images")
configure_file(paths_config.h.in "${PROJECT_BINARY_DIR}/generated/utils/cuda/paths_config.h" @ONLY)
dehancer_add_cuda_shaders(UtilsKernels_cuda "${CMAKE_CURRENT_SOURCE_DIR}/TestKernels.cu")
