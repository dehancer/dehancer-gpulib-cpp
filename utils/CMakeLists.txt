set (CURRENT_KERNEL_TARGET_NAME UtilsKernels)

if (DEHANCER_GPU_METAL)
    set (DEHANCER_CURRENT_PLATFORM_METAL ON)
endif ()

if (DEHANCER_GPU_CUDA)
    set (DEHANCER_CURRENT_PLATFORM_CUDA ON)
endif ()

if (DEHANCER_GPU_OPENCL)
    set (DEHANCER_CURRENT_PLATFORM_OPENCL ON)
endif ()

add_subdirectory(shaders)
add_subdirectory(ocio_codegen)
