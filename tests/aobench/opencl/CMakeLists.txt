set (TEST opencl-aobench-test)

include(${clHelper_DIR}/clHelper.cmake)

include_directories(
        ${CMAKE_SOURCE_DIR}/tests/shaders
)

OPENCL_INCLUDE_DIRECTORIES(
        ${CMAKE_SOURCE_DIR}/tests/shaders
        .
)

COMPILE_OPENCL(
        exampleKernel.cl
)

include(${CMAKE_SOURCE_DIR}/tests/CMakeCommon.in)