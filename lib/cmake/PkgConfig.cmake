function(dehancer_install_pkg_config target backend)
    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(prefix "${CMAKE_INSTALL_PREFIX}")
    else()
        file(RELATIVE_PATH pc_prefix "/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "/")
        set(prefix "\${pcfiledir}/${pc_prefix}")
    endif()

    if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(libdir "${CMAKE_INSTALL_LIBDIR}")
    else()
        set(libdir "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
    endif()

    if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(includedir "${CMAKE_INSTALL_INCLUDEDIR}")
    else()
        set(includedir "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    endif()

    get_target_property(dehancer_gpu_cpp_lib ${target} OUTPUT_NAME)

    if(NOT dehancer_gpu_cpp_lib)
        set(dehancer_gpu_cpp_lib "${target}")
    endif()

    set(requires "dehancer-common-cpp dehancer-xmp-cpp dehancer-maths-cpp")
    set(requires_private "opencv4")
    set(libs_private "${CMAKE_THREAD_LIBS_INIT}")
    set(cflags "")

    get_target_property(definitions ${target} INTERFACE_COMPILE_DEFINITIONS)

    foreach(definition IN LISTS definitions)
        string(APPEND cflags " -D${definition}")
    endforeach()

    if(backend STREQUAL "metal")
        string(APPEND libs_private " -framework Metal -framework MetalKit -framework MetalPerformanceShaders -framework CoreImage -framework Foundation -framework CoreVideo")

        if(IOS)
            string(APPEND libs_private " -framework UIKit")
        else()
            string(APPEND libs_private " -framework Cocoa -framework IOKit")
        endif()

    elseif(backend STREQUAL "opencl")
        # The helper installs a library and headers, but no pkg-config file.
        string(APPEND libs_private " -lclHelperLib")
        string(APPEND cflags " -DCL_TARGET_OPENCL_VERSION=120")

        if(APPLE)
            string(APPEND libs_private " -framework OpenCL -Wl,-export_dynamic")
        else()
            string(APPEND libs_private " -lOpenCL")
            # Bear in mind that CMake's `BSD` excludes Darwin, so this will not match macOS.
            # Happy building on 386BSD!
            if(LINUX OR BSD)
                string(APPEND libs_private " -Wl,--export-dynamic")
            endif()
        endif()

        foreach(library IN LISTS CMAKE_DL_LIBS)
            string(APPEND libs_private " -l${library}")
        endforeach()

    elseif(backend STREQUAL "cuda")
        foreach(directory IN LISTS CUDAToolkit_INCLUDE_DIRS)
            string(APPEND cflags " -I\"${directory}\"")
        endforeach()

        if(CUDAToolkit_LIBRARY_DIR)
            string(APPEND libs_private " -L\"${CUDAToolkit_LIBRARY_DIR}\"")
        endif()

        string(APPEND libs_private " -lcudart")
        if(WIN32)
            string(APPEND libs_private " -lnvcuda")
        else()
            string(APPEND libs_private " -lcuda")
        endif()
    endif()

    string(STRIP "${libs_private}" libs_private)

    configure_file("${PROJECT_SOURCE_DIR}/dehancer-gpulib-cpp.pc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/dehancer-gpulib-cpp.pc" @ONLY
    )

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dehancer-gpulib-cpp.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    )
endfunction()
