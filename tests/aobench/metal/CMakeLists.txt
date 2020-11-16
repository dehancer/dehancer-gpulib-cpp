
#
# Generators
#
set(METAL_KERNELS_LIBRARY ${CMAKE_BINARY_DIR}/tests/shaders/metal/TestKernels.metallib)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/paths_config.h.in ${CMAKE_BINARY_DIR}/generated/tests/paths_config.h)

set (TEST metal-aobench-test)
include(${CMAKE_SOURCE_DIR}/tests/CMakeCommon.in)

#add_dependencies(metal-aobench-test TestKernels.metallib)