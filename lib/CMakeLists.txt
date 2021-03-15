include(CMakePackageConfigHelpers)

set (PUBLIC_INCLUDE_FILES ${CMAKE_SOURCE_DIR}/include/dehancer)
set (EMBEDDED_IMAGES_DIR ${CMAKE_BINARY_DIR}/generated/embedded)

file (GLOB COMMON_SRC ${COMMON_SRC}
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.c
        ${CMAKE_SOURCE_DIR}/src/math/*.cpp
        ${CMAKE_SOURCE_DIR}/src/ocio/*.cpp
        ${CMAKE_SOURCE_DIR}/src/operations/*.cpp
        ${CMAKE_SOURCE_DIR}/src/overlays/*.cpp
        ${CMAKE_SOURCE_DIR}/src/platforms/*.cpp
        )

include_directories(
        ${CMAKE_SOURCE_DIR}/src
)

if (DEHANCER_GPU_METAL)
    add_subdirectory(metal)
endif()

if(DEHANCER_GPU_OPENCL)
    add_subdirectory(opencl)
endif ()

if(DEHANCER_GPU_CUDA)
    add_subdirectory(cuda)
endif ()


install(DIRECTORY ${PUBLIC_INCLUDE_FILES}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
        )