if (DEHANCER_GPU_METAL)
    set (DEHANCER_CURRENT_PLATFORM_METAL ON)
endif ()

if (DEHANCER_GPU_CUDA)
    set (DEHANCER_CURRENT_PLATFORM_CUDA ON)
endif ()

if (DEHANCER_GPU_OPENCL)
    set (DEHANCER_CURRENT_PLATFORM_OPENCL ON)
endif ()

set (TEST common-w-cache-test)
include(${CMAKE_SOURCE_DIR}/tests/CMakeCommon.in)
