function(dehancer_add_test name)
    file(GLOB sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
    )
    add_executable(${name} ${sources})
    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}"
        "${PROJECT_BINARY_DIR}/generated"
        "${PROJECT_BINARY_DIR}/tests/generated"
    )
    target_link_libraries(${name} PRIVATE
        dehancer_gpulib::dehancer_gpulib_${DEHANCER_BACKEND_GPU_NAME}
        GTest::gtest_main
        ${OpenCV_LIBS}
    )
    set_target_properties(${name} PROPERTIES
        OBJCXX_STANDARD 17
        OBJCXX_STANDARD_REQUIRED ON
    )
    if(DEHANCER_GPU_METAL)
        add_dependencies(${name} TestKernels_metal)
    else()
        target_link_libraries(${name} PRIVATE TestKernels_${DEHANCER_BACKEND_GPU_NAME})
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()
