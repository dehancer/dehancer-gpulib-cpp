include(CMakePackageConfigHelpers)

#set(lock "${CMAKE_BINARY_DIR}/lib-lock.lock")
#string(TOLOWER ${lock} lock)

#file(LOCK ${lock} TIMEOUT 10 RESULT_VARIABLE lock_result)

if (NOT ${lock_result} EQUAL 0)
    message(FATAL_ERROR "DEADLOCKED: ${lock_result}")
endif()

set (PUBLIC_INCLUDE_FILES ${DEHANCER_GPULIB_CPP_SOURCE}/include/dehancer)
set (EMBEDDED_IMAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/embedded)

file (GLOB COMMON_SRC ${COMMON_SRC}
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/*.c
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/spaces/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/clut/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/clut/utils/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/profile/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/math/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/ocio/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/operations/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/overlays/*.cpp
        ${DEHANCER_GPULIB_CPP_SOURCE}/src/platforms/*.cpp
        )

include_directories(
        ${CMAKE_SOURCE_DIR}/src
)

if (DEHANCER_GPU_METAL)
    add_subdirectory(metal)
endif()

if(DEHANCER_GPU_OPENCL)
    add_definitions(-DDEHANCER_GPU_OPENCL=1)
    add_subdirectory(opencl)
endif ()

if(DEHANCER_GPU_CUDA)
    add_subdirectory(cuda)
endif ()
