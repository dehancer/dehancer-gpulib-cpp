
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEHANCER_GPU_CUDA=1")

set (PROJECT_LIB_PLATFORM_SUFFIX _cuda)
set (PROJECT_LIB_PLATFORM ${PROJECT_LIB}${PROJECT_LIB_PLATFORM_SUFFIX})

file (GLOB COMMON_IMPL_SRC ${COMMON_IMPL_SRC}
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/platforms/cuda/*.cpp
)

include(${DEHANCER_GPULIB_CPP_SOURCE}/lib/CMake.in)