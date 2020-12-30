set (CURRENT_KERNEL_TARGET_NAME TestKernels)


if (DEHANCER_GPU_CUDA)
    add_subdirectory(cuda)
endif ()

add_subdirectory(shaders)
add_subdirectory(aobench)
add_subdirectory(device)
add_subdirectory(blur)
add_subdirectory(texture)