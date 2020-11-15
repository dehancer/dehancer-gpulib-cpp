if (DEHANCER_GPU_METAL)
    add_subdirectory(metal)
elseif(DEHANCER_GPU_OPENCL)
    add_subdirectory(opencl)
endif ()

