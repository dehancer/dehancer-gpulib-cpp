set (CURRENT_KERNEL_TARGET_NAME TestKernels)

set(IMAGES_DIR ${DEHANCER_GPULIB_CPP_SOURCE}/tests/images)
set(DATA_DIR ${DEHANCER_GPULIB_CPP_SOURCE}/tests/data)
configure_file(${DEHANCER_GPULIB_CPP_SOURCE}/tests/include/test_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/tests/test_config.h)

include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

if (DEHANCER_GPU_CUDA)
    add_definitions("-DDEHANCER_GPU_CUDA=1")
    add_subdirectory(cuda)
endif ()

if (DEHANCER_GPU_METAL)
    add_definitions("-DDEHANCER_GPU_METAL=1")
endif ()

if (DEHANCER_GPU_OPENCL)
    add_definitions("-DDEHANCER_GPU_OPENCL=1")
endif ()

add_subdirectory(shaders)
add_subdirectory(aobench)
add_subdirectory(device)
add_subdirectory(texture)
add_subdirectory(video)
add_subdirectory(function)
add_subdirectory(memory)
add_subdirectory(operations)
add_subdirectory(filter)
add_subdirectory(overlays)
add_subdirectory(clut)
add_subdirectory(histogram)