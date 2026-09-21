set(METAL_KERNELS_LIBRARY "${PROJECT_BINARY_DIR}/tests/shaders/metal/TestKernels_metal.metallib")
configure_file(paths_config.h.in "${PROJECT_BINARY_DIR}/generated/tests/paths_config.h" @ONLY)

dehancer_add_test(metal-aobench-test)

add_custom_command(TARGET metal-aobench-test POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${METAL_KERNELS_LIBRARY}"
        "$<TARGET_FILE_DIR:metal-aobench-test>/"
    VERBATIM
)
