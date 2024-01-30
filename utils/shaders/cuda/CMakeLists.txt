set(CMAKE_CXX_STANDARD 14)

set(CURRENT_TARGET_NAME "${CURRENT_KERNEL_TARGET_NAME}_cuda")
set(CURRENT_TARGET_LIB ${CURRENT_TARGET_NAME}.fatbin)

set(CUDA_KERNELS_LIBRARY ${CMAKE_BINARY_DIR}/tests/shaders/cuda/${CURRENT_TARGET_LIB})

set(CUDA_IMAGE_DIR ${CMAKE_SOURCE_DIR}/tests/images)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/paths_config.h.in ${CMAKE_BINARY_DIR}/generated/tests/cuda/paths_config.h)

include_directories(
        ${CMAKE_SOURCE_DIR}/tests/shaders
        ${CMAKE_SOURCE_DIR}/include
        .
)

file (GLOB DEPENDS_SOURCES ${DEPENDS_PLUGIN_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_SOURCE_DIR}/tests/shaders/*.h
        )

add_custom_target(
        ${CURRENT_TARGET_NAME}
        DEPENDS ${DEPENDS_SOURCES}
        COMMAND ${CMAKE_CUDA_COMPILER}
        -Xcompiler ${CMAKE_CUDA_COMPILER_FLAGS}
        -ccbin ${CUDA_BIN_COMPILER} -m64
        --gpu-architecture=all
        ${CUDA_EXTRA_ARCH}
        --use_fast_math
        -ftz=true -prec-div=false -prec-sqrt=false
        --extra-device-vectorization
        --keep-device-functions
        -O3
        -Wno-deprecated-declarations
        -Wno-deprecated-gpu-targets
        -o ${CURRENT_TARGET_LIB}
        -fatbin ${KERNEL_SOURCE}
        -I${CMAKE_BINARY_DIR}/external/include
        -I${CMAKE_SOURCE_DIR}/framework/include/dehancer/framework
        -I${CMAKE_SOURCE_DIR}/framework
        -I${CMAKE_CURRENT_SOURCE_DIR}
        ${GPU_INCLUDE_DIRS}
        -DCUDA_KERNEL=1
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
