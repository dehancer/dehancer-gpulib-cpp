set (DEHANCER_CURRENT_PLATFORM_METAL ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEHANCER_GPU_METAL=1")

set (UTIL_NAME dh-ocio-codegen)
include(${CMAKE_SOURCE_DIR}/utils/CMakeCommon.in)