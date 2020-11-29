if (DEHANCER_GPU_METAL)
    #add_subdirectory(metal)
endif()

if(DEHANCER_GPU_OPENCL)
    add_subdirectory(opencl)
endif ()

