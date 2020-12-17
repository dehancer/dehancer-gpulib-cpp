set(CURRENT_TARGET_NAME "${CURRENT_KERNEL_TARGET_NAME}_cuda")
set(CURRENT_TARGET_LIB ${CURRENT_TARGET_NAME}.fatbin)

set(CUDA_KERNELS_LIBRARY ${CMAKE_BINARY_DIR}/tests/shaders/cuda/${CURRENT_TARGET_LIB})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/paths_config.h.in ${CMAKE_BINARY_DIR}/generated/tests/cuda/paths_config.h)

include_directories(
        ${CMAKE_SOURCE_DIR}/tests/shaders
        .
)

file (GLOB DEPENDS_SOURCES ${DEPENDS_PLUGIN_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_SOURCE_DIR}/tests/shaders/*.h
        )

add_custom_target(
        ${CURRENT_TARGET_NAME}
        DEPENDS
        ${DEPENDS_SOURCES}
        COMMAND ${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_FLAGS}
        -ccbin ${CUDA_BIN_COMPILER} -m64 -gencode arch=compute_35,code=compute_35 -o ${CURRENT_TARGET_LIB}
        -fatbin ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}.cu
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES BUNDLE TRUE)
set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES PREFIX "")
set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES SUFFIX ".ptx")

