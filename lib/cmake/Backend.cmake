include("${CMAKE_CURRENT_LIST_DIR}/EmbedOverlays.cmake")

function(dehancer_add_backend backend)
    file(GLOB COMMON_SRC CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.c"
        "${PROJECT_SOURCE_DIR}/src/spaces/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/clut/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/clut/utils/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/profile/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/math/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/ocio/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/operations/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/overlays/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/platforms/*.cpp"
    )

    set(target "${PROJECT_NAME}_${backend}")
    string(TOUPPER "${backend}" backend_upper)
    file(GLOB COMMON_IMPL_SRC CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/platforms/${backend}/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/platforms/${backend}/*.mm"
    )
    dehancer_embed_overlays(embedded_sources)
    add_library(${target} STATIC
        ${COMMON_SRC} ${COMMON_IMPL_SRC} ${embedded_sources}
    )
    add_library(dehancer_gpulib::${target} ALIAS ${target})
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OBJCXX_STANDARD 17
        OBJCXX_STANDARD_REQUIRED ON
    )
    if(IOS AND SDK_NAME)
        set_target_properties(${target} PROPERTIES OUTPUT_NAME "${target}_${SDK_NAME}")
    endif()
    target_include_directories(${target}
        PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
            "${PROJECT_SOURCE_DIR}"
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_BINARY_DIR}/generated"
    )
    target_link_libraries(${target}
        PUBLIC
            dehancer_maths_cpp::dehancer_maths_cpp
            dehancer_common_cpp::dehancer_common_cpp
            dehancer_xmp_cpp::dehancer_xmp_cpp
        PRIVATE Threads::Threads ${OpenCV_LIBS}
    )
    target_compile_definitions(${target} PUBLIC DEHANCER_GPU_${backend_upper}=1)
    foreach(flag IN ITEMS PRINT_DEBUG PRINT_KERNELS_DEBUG DEHANCER_OPENCL_CONTEXT_NOT_RELEASE)
        if(${flag})
            target_compile_definitions(${target} PRIVATE ${flag}=1)
        endif()
    endforeach()
    if(MSVC)
        target_compile_definitions(${target} PUBLIC _USE_MATH_DEFINES CV_IGNORE_DEBUG_BUILD_GUARD=1)
        target_compile_options(${target} PRIVATE /EHsc /GR)
    else()
        target_compile_options(${target} PRIVATE -Wno-unused-parameter)
    endif()
    if(DEHANCER_DEBUG)
        target_compile_definitions(${target} PRIVATE DEBUG=1)
        if(MSVC)
            target_compile_options(${target} PRIVATE /Z7 /UNDEBUG)
        else()
            target_compile_options(${target} PRIVATE -g -UNDEBUG)
        endif()
    endif()
    if(LINUX)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    endif()
    if(IOS)
        target_compile_definitions(${target} PUBLIC IOS_SYSTEM=13)
    endif()
    if(backend STREQUAL "metal")
        target_compile_options(${target} PRIVATE -fno-objc-arc)
        foreach(framework IN ITEMS Metal MetalKit MetalPerformanceShaders CoreImage Foundation CoreVideo)
            target_link_libraries(${target} PRIVATE "-framework ${framework}")
        endforeach()
        if(IOS)
            target_link_libraries(${target} PRIVATE "-framework UIKit")
        else()
            target_link_libraries(${target} PRIVATE "-framework Cocoa" "-framework IOKit")
        endif()
        if(DEHANCER_USE_NATIVE_APPLE_API)
            target_compile_definitions(${target} PUBLIC DEHANCER_USE_NATIVE_APPLE_API=1)
        endif()
    elseif(backend STREQUAL "opencl")
        target_link_libraries(${target} PUBLIC dehancer_opencl_helper::dehancer_opencl_helper)
        target_compile_definitions(${target} PRIVATE "CL_BUILD_PROGRAM_ARGS=\"${CL_BUILD_PROGRAM_ARGS}\"")
    elseif(backend STREQUAL "cuda")
        target_link_libraries(${target} PUBLIC CUDA::cudart CUDA::cuda_driver)
    endif()
    install(TARGETS ${target} EXPORT dehancer_gpulibTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
    if(CREATE_PKG_CONFIG)
        include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PkgConfig.cmake")
        dehancer_install_pkg_config(${target} "${backend}")
    endif()
endfunction()
