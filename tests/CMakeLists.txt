set (CURRENT_KERNEL_TARGET_NAME TestKernels)

set(IMAGES_DIR ${CMAKE_SOURCE_DIR}/tests/images)
configure_file(${CMAKE_SOURCE_DIR}/tests/include/test_config.h.in ${CMAKE_BINARY_DIR}/generated/tests/test_config.h)


if (DEHANCER_GPU_CUDA)
    add_subdirectory(cuda)
endif ()

add_subdirectory(shaders)
add_subdirectory(aobench)
add_subdirectory(device)
add_subdirectory(texture)
add_subdirectory(function)
add_subdirectory(memory)
add_subdirectory(operations)