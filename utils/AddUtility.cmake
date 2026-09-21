function(dehancer_add_utility name)
    file(GLOB sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.mm"
    )
    add_executable(${name} ${sources})
    target_include_directories(${name} PRIVATE "${PROJECT_BINARY_DIR}/generated")
    target_link_libraries(${name} PRIVATE dehancer_gpulib::dehancer_gpulib_metal)
    add_dependencies(${name} UtilsKernels_metal)
endfunction()
