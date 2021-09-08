set (CURRENT_KERNEL_TARGET_NAME UtilsKernels)

if (DEHANCER_GPU_METAL)
    set (DEHANCER_CURRENT_PLATFORM_METAL ON)

    ##
    ## TODO: OPENCL/Cuda version
    ##
    add_subdirectory(shaders)
    add_subdirectory(ocio_codegen)
    add_subdirectory(lut_codegen)

endif ()

if (DEHANCER_GPU_CUDA)
    set (DEHANCER_CURRENT_PLATFORM_CUDA ON)
endif ()

if (DEHANCER_GPU_OPENCL)
    set (DEHANCER_CURRENT_PLATFORM_OPENCL ON)
endif ()

