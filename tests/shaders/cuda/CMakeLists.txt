set(CMAKE_CXX_STANDARD 14)

set(CURRENT_TARGET_NAME "${CURRENT_KERNEL_TARGET_NAME}_cuda")
set(CURRENT_TARGET_LIB ${CURRENT_TARGET_NAME}.fatbin)
set(CURRENT_TARGET_EMBEDDED ${CURRENT_TARGET_NAME}.c)

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
        -ccbin ${CUDA_BIN_COMPILER}
        -m64 -ptx
        --use_fast_math
        -ftz=true -prec-div=false -prec-sqrt=false
        --extra-device-vectorization
        --keep-device-functions
        -Wno-deprecated-declarations
        -Xptxas -v
        -o ${CURRENT_TARGET_LIB}
        -fatbin ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}.cu
        -I${CMAKE_SOURCE_DIR}/include
        -I${CMAKE_SOURCE_DIR}/tests/shaders
        -DCUDA_KERNEL=1
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
        TARGET ${CURRENT_TARGET_NAME} POST_BUILD
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND xxd
        -i ${CURRENT_TARGET_LIB}
        ${CURRENT_TARGET_EMBEDDED}
        DEPENDS ${DEPENDS_SOURCES} ${CURRENT_TARGET_NAME}
        COMMENT "embedding opencl code from ${CURRENT_TARGET_LIB} ->  ${CURRENT_TARGET_EMBEDDED}"
)