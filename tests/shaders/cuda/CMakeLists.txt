set(CMAKE_CXX_STANDARD 14)

set(CURRENT_TARGET_NAME "${CURRENT_KERNEL_TARGET_NAME}_cuda")
set(CURRENT_TARGET_LIB ${CURRENT_TARGET_NAME}.fatbin)

set(CURRENT_TARGET_NAME2 "${CURRENT_KERNEL_TARGET_NAME}2_cuda")
set(CURRENT_TARGET_LIB2 ${CURRENT_TARGET_NAME2}.fatbin)

set(CUDA_KERNELS_LIBRARY ${CMAKE_BINARY_DIR}/tests/shaders/cuda/${CURRENT_TARGET_LIB})
set(CUDA_KERNELS_LIBRARY2 ${CMAKE_BINARY_DIR}/tests/shaders/cuda/${CURRENT_TARGET_LIB2})

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
        ${CMAKE_CUDA_COMPILER_FLAGS}
        -std=c++14
        -ccbin ${CUDA_BIN_COMPILER} -m64 -gencode arch=compute_35,code=compute_35
        -o ${CURRENT_TARGET_LIB}
        -fatbin ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}.cu
        -I${CMAKE_SOURCE_DIR}/include
        -I${CMAKE_SOURCE_DIR}/tests/shaders
        -DCUDA_KERNEL=1
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(
        ${CURRENT_TARGET_NAME2}
        DEPENDS ${DEPENDS_SOURCES}
        COMMAND ${CMAKE_CUDA_COMPILER}
        ${CMAKE_CUDA_COMPILER_FLAGS}
        -std=c++14
        -ccbin ${CUDA_BIN_COMPILER} -m64 -gencode arch=compute_35,code=compute_35
        -o ${CURRENT_TARGET_LIB2}
        -fatbin ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}2.cu
        -I${CMAKE_SOURCE_DIR}/include
        -I${CMAKE_SOURCE_DIR}/tests/shaders
        -DCUDA_KERNEL=1
        #-O2 --use_fast_math

        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

#set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES BUNDLE TRUE)
#set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES PREFIX "")
#set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES SUFFIX ".fatbin")
#
#set_target_properties(${CURRENT_TARGET_NAME2} PROPERTIES BUNDLE TRUE)
#set_target_properties(${CURRENT_TARGET_NAME2} PROPERTIES PREFIX "")
#set_target_properties(${CURRENT_TARGET_NAME2} PROPERTIES SUFFIX ".fatbin")
