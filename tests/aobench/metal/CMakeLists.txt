#
# Generators
#
set(METAL_KERNELS_LIBRARY ${CMAKE_BINARY_DIR}/tests/shaders/metal/TestKernels.metallib)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/paths_config.h.in ${CMAKE_BINARY_DIR}/generated/tests/paths_config.h)

set (TEST metal-aobench-test)
set (DEHANCER_CURRENT_PLATFORM_METAL ON)
include(${CMAKE_SOURCE_DIR}/tests/CMakeCommon.in)

add_custom_command(
        TARGET ${TEST} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/tests/shaders/metal/TestKernels.metallib
        "$<TARGET_FILE_DIR:${TEST}>/"
)