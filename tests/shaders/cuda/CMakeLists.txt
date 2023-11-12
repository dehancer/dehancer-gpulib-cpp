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
#        --generate-code arch=compute_35,code=sm_35
#        --generate-code arch=compute_50,code=sm_50
#        --generate-code arch=compute_52,code=sm_52
#        --generate-code arch=compute_53,code=sm_53
#        --generate-code arch=compute_60,code=sm_60
#        --generate-code arch=compute_61,code=sm_61
#        --generate-code arch=compute_62,code=sm_62
#        --generate-code arch=compute_70,code=sm_70
        --generate-code arch=compute_72,code=sm_72
        --generate-code arch=compute_75,code=sm_75
        --generate-code arch=compute_80,code=sm_80
        --generate-code arch=compute_86,code=sm_86
        --generate-code arch=compute_87,code=sm_87
        --generate-code arch=compute_89,code=sm_89
        --generate-code arch=compute_90,code=sm_90

        --use_fast_math
        -ftz=true -prec-div=false -prec-sqrt=false
        --extra-device-vectorization
        --keep-device-functions
        -Wno-deprecated-declarations
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