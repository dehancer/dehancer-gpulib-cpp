set(CUDA_KERNELS_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/TestKernels.fatbin")
set(CUDA_IMAGE_DIR "${PROJECT_SOURCE_DIR}/tests/images")
configure_file(paths_config.h.in "${PROJECT_BINARY_DIR}/generated/tests/cuda/paths_config.h" @ONLY)
dehancer_add_cuda_shaders(TestKernels "${CMAKE_CURRENT_SOURCE_DIR}/TestKernels.cu")
