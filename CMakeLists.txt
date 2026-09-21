cmake_minimum_required(VERSION 4.3)
project(dehancer_gpulib VERSION 1.13.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(CREATE_PKG_CONFIG "Generate and install pkg-config metadata" OFF)
option(BUILD_TESTING "Build tests" OFF)

option(DEHANCER_GPU_METAL "Build Metal backend" OFF)
option(DEHANCER_GPU_OPENCL "Build OpenCL backend" OFF)
option(DEHANCER_GPU_CUDA "Build CUDA backend" OFF)

option(DEHANCER_USE_NATIVE_APPLE_API "Use native Apple image APIs" ON)
option(DEHANCER_OPENCL_CONTEXT_NOT_RELEASE "Keep OpenCL contexts alive" OFF)
option(PRINT_DEBUG "Enable debug logging" ON)
option(PRINT_KERNELS_DEBUG "Enable kernel logging" OFF)
option(DEHANCER_DEBUG "Enable additional debug information" OFF)
option(USE_OPENCOLORIO "Build OpenColorIO utilities" OFF)
option(BUILD_UTILS "Build utilities" OFF)
set(CL_STD_VERSION "2.0" CACHE STRING "OpenCL kernel standard") # FIXME maybe 3.0?
set(CL_BUILD_PROGRAM_ARGS "-cl-std=CL${CL_STD_VERSION} -cl-kernel-arg-info -cl-unsafe-math-optimizations -cl-single-precision-constant" CACHE STRING "OpenCL kernel compiler arguments")

set(DEHANCER_GPULIB_VERSION 1) # FIXME what for?

foreach(dependency IN ITEMS dehancer_common_cpp dehancer_xmp_cpp dehancer_maths_cpp)
    if(NOT TARGET ${dependency}::${dependency})
        find_package(${dependency} CONFIG REQUIRED)
    endif()
endforeach()

find_package(Threads REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

if(DEHANCER_GPU_METAL)
    set(DEHANCER_BACKEND metal)
    if(NOT APPLE)
        message(FATAL_ERROR "Metal requires an Apple platform")
    endif()
    enable_language(OBJCXX)

elseif(DEHANCER_GPU_OPENCL)
    set(DEHANCER_BACKEND opencl)
    if(NOT TARGET dehancer_opencl_helper::dehancer_opencl_helper)
        find_package(dehancer_opencl_helper CONFIG REQUIRED)
    endif()

elseif(DEHANCER_GPU_CUDA)
    set(DEHANCER_BACKEND cuda)
    find_package(CUDAToolkit REQUIRED)

else()
    message(FATAL_ERROR "Enable a DEHANCER_GPU backend: METAL, OPENCL, or CUDA")
endif()

if(USE_OPENCOLORIO)
    find_package(OpenColorIO CONFIG REQUIRED)
endif()

configure_file(include/dehancer/gpulib_version.h.in generated/gpulib_version.h @ONLY) # FIXME what for?

add_subdirectory(lib)

set(package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/dehancer_gpulib")
configure_package_config_file(lib/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/dehancer_gpulibConfig.cmake"
    INSTALL_DESTINATION "${package_dir}"
)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/dehancer_gpulibConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)
install(EXPORT dehancer_gpulibTargets
    NAMESPACE dehancer_gpulib::
    DESTINATION "${package_dir}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dehancer_gpulibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dehancer_gpulibConfigVersion.cmake"
    DESTINATION "${package_dir}"
)
install(DIRECTORY include/dehancer DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/gpulib_version.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dehancer"
)

if(BUILD_UTILS OR BUILD_TESTING)
    include(lib/cmake/Shaders.cmake)
endif()

if(BUILD_UTILS AND NOT IOS)
    add_subdirectory(utils)
endif()

if(BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    include(CTest)
    add_subdirectory(tests)
endif()
