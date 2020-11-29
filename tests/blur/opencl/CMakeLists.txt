set (TEST opencl-gaussian-blur-test)
set (DEHANCER_CURRENT_PLATFORM_OPENCL ON)

include(${clHelper_DIR}/clHelper.cmake)

include_directories(
        ${CMAKE_SOURCE_DIR}/tests/shaders
)

OPENCL_INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests/shaders
        .
)

COMPILE_OPENCL(
        blurKernel.cl
)

include(${CMAKE_SOURCE_DIR}/tests/CMakeCommon.in)