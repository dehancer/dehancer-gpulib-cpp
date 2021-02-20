
if (DEHANCER_GPU_OPENCL)
    add_subdirectory(opencl)
endif()

if (DEHANCER_GPU_CUDA)
    add_subdirectory(cuda)
endif()

if (DEHANCER_GPU_METAL)
    add_subdirectory(metal)
endif()
