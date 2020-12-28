
if (DEHANCER_GPU_CUDA)
    add_subdirectory(opencl)
endif()

if (DEHANCER_GPU_CUDA)
    add_subdirectory(cuda)
endif()