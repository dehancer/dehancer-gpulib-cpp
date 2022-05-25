set (CURRENT_KERNEL_TARGET_NAME UtilsKernels)

if (DEHANCER_GPU_METAL)
    set (DEHANCER_CURRENT_PLATFORM_METAL ON)

    add_subdirectory(shaders)

    if (BUILD_UTILS)
        if (USE_OPENCOLORIO)
            add_subdirectory(ocio_codegen)
        endif ()
        add_subdirectory(lut_codegen)
    endif ()

endif ()

if (DEHANCER_GPU_CUDA)
    set (DEHANCER_CURRENT_PLATFORM_CUDA ON)
endif ()

if (DEHANCER_GPU_OPENCL)
    set (DEHANCER_CURRENT_PLATFORM_OPENCL ON)
endif ()

