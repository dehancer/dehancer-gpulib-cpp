#
# Generators
#
set(METAL_KERNELS_LIBRARY ${CMAKE_BINARY_DIR}/tests/shaders/metal/${CURRENT_KERNEL_TARGET_NAME}_metal.metallib)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/paths_config.h.in ${CMAKE_BINARY_DIR}/generated/tests/paths_config.h)

set (TEST metal-aobench-test)
set (DEHANCER_CURRENT_PLATFORM_METAL ON)
include(${DEHANCER_GPULIB_CPP_SOURCE}/tests/CMakeCommon.in)

add_custom_command(
        TARGET ${TEST} POST_BUILD
        DEPENDS TestKernels
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/tests/shaders/metal/${CURRENT_KERNEL_TARGET_NAME}_metal.metallib
        "$<TARGET_FILE_DIR:${TEST}>/"
)