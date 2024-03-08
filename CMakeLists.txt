cmake_minimum_required(VERSION 3.22)


#
#  Current master version release dependencies:
#  dehancer-common-cpp:0.14.0
#     base64cpp:1.2.0
#     ed25519cpp:1.4.0
#  dehancer-xmp-cpp:0.10.0
#  dehancer-maths-cpp:0.8.0
#  dehancer-opencl-helper:0.1.0
#

set(GIT_TAG_dehancer_xmp_cpp "develop")
set(GIT_TAG_dehancer_maths_cpp "develop")

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(GIT_TAG_dehancer_xmp_cpp "0.10.0")
    set(GIT_TAG_dehancer_maths_cpp "0.8.0")
endif ()

#
# Project
#

set(PROJECT_LIB dehancer_gpulib)
set(DEHANCER_GPULIB_VERSION 1)
project(${PROJECT_LIB} VERSION 1.7.0)

#
#
# C++
#
set(CMAKE_CXX_STANDARD 17)


if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

#
# Options
#
option (BUILD_TESTING "Enable creation of  tests." OFF)
option (BUILD_TESTING_GPULIB "Override BUILD_TESTING for GPULIB" ON)
option (DEHANCER_GPU_METAL "Dehancer GPU Layer is Metal" ON)
option (DEHANCER_GPU_OPENCL "Dehancer GPU Layer is OpenCL" ON)
option (DEHANCER_GPU_CUDA "Dehancer GPU Layer is CUDA" ON)
option (VCPKG_TARGET_TRIPLET "VCPKG_TARGET_TRIPLET" "")
option (PRINT_KERNELS_DEBUG "PRINT_KERNELS_DEBUG" OFF)
option (PRINT_DEBUG "PRINT_DEBUG" ON)
option (DEHANCER_DEBUG "DEHANCER_DEBUG" OFF)
option (USE_OPENCOLORIO "USE_OPENCOLORIO" OFF)
option (DEHANCER_USE_NATIVE_APPLE_API "DEHANCER_USE_NATIVE_APPLE_API" ON)
option (BUILD_UTILS "BUILD_UTILS" OFF)
option (DEHANCER_OPENCL_CONTEXT_NOT_RELEASE "DEHANCER_OPENCL_CONTEXT_NOT_RELEASE" OFF)
option (DEHANCER_CONTROLLED_SINGLETON "DEHANCER_CONTROLLED_SINGLETON" OFF)
option (DEHANCER_INSTALL_COMMON_DEPS "DEHANCER_INSTALL_COMMON_DEPS" ON)
option (OpenCV_DIR "OpenCV Root" "/usr/local/universal")
option (GTest_ROOT "GTest_ROOT" "")

set(DEHANCER_GPULIB_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(DEHANCER_GPULIB_BINARY ${CMAKE_CURRENT_BINARY_DIR})

if (NOT OpenCV_DIR)
    if (APPLE)
        set (OpenCV_DIR "/usr/local/universal" CACHE INTERNAL "OpenCV Root" )
    elseif (LINUX)
        set (OpenCV_DIR "/usr/local" CACHE INTERNAL "OpenCV Root" )
    else ()
        set (OpenCV_DIR "" CACHE INTERNAL "OpenCV Root" )
    endif ()
endif ()

if (NOT BUILD_TESTING_GPULIB)
    set (BUILD_TESTING OFF CACHE INTERNAL "-")
endif ()


message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "GTest_ROOT: ${GTest_ROOT}")

message(STATUS "VCPKG_TARGET_TRIPLET: ${VCPKG_TARGET_TRIPLET}")
message(STATUS "DEHANCER_DEBUG: ${DEHANCER_DEBUG}")
message(STATUS "PRINT_DEBUG: ${PRINT_DEBUG}")
message(STATUS "DEBUG: ${DEBUG}")
message(STATUS "PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
message(STATUS "APPLE: ${APPLE}")
message(STATUS "LINUX: ${LINUX}")
message(STATUS "XCODE_VERSION: ${XCODE_VERSION}")
message(STATUS "SDK_NAME: ${SDK_NAME}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "APPLE_TARGET_TRIPLE: ${APPLE_TARGET_TRIPLE}")
message(STATUS "DEPLOYMENT_TARGET: ${DEPLOYMENT_TARGET}")
message(STATUS "DEHANCER_INSTALL_COMMON_DEPS: ${DEHANCER_INSTALL_COMMON_DEPS}")


if(DEHANCER_OPENCL_CONTEXT_NOT_RELEASE)
    add_definitions(-DDEHANCER_OPENCL_CONTEXT_NOT_RELEASE=1)
endif ()

if(DEHANCER_CONTROLLED_SINGLETON)
    add_definitions(-DDEHANCER_CONTROLLED_SINGLETON=1)
endif()

# googletest, etc
set (EXTERNAL_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external)
set (EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
set (EXTERNAL_INCLUDE_DIR ${EXTERNAL_INSTALL_LOCATION}/include)
set (COMMON_DEPENDENCIES ${CURRENT_METAL_TARGET_NAME})
set (COMMON_LIBRARIES)

if (NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif ()

#
# Programs
#
find_program(CCACHE_FOUND ccache)

find_package (Threads REQUIRED)
find_package (PkgConfig REQUIRED)
find_package (EXPAT REQUIRED)
find_package (Iconv REQUIRED)
find_package (ZLIB REQUIRED)

#
# CCache settings
#
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

if(LINUX)
    set(STATIC_LINKING OFF)
    message(STATUS "Linux configuration...")
    if (DEHANCER_GPU_METAL)
        message("Linux GPU Layer does not support Metal SDK")
        set(DEHANCER_GPU_METAL OFF)
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -W -Wall -Wextra -Wno-ignored-qualifiers -Wvariadic-macros")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -Wno-pedantic -Wunused-function")
elseif(APPLE)

    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set (IOS TRUE)
        add_definitions(-DIOS_SYSTEM=13)
    else()
        set (MACOS TRUE)
    endif ()

    if(CMAKE_OSX_ARCHITECTURES)
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
            add_definitions(-DDEHANCER_ARCH="arm64" -DDEHANCER_ARCH_IS_ARM64=1)
        else()
            add_definitions(-DDEHANCER_ARCH="x86_64" -DDEHANCER_ARCH_IS_X86_64=1)
            add_definitions(-DDEHANCER_ARCH_MACOS=1)
        endif ()
    endif()

    set(STATIC_LINKING ON)
    message(STATUS "Apple configuration...")
    if (DEHANCER_GPU_METAL)
        message(STATUS "Apple uses Metal...")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ObjC++ -fno-objc-arc -Wno-unused-command-line-argument -F/Library/Frameworks -Wno-unqualified-std-cast-call")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Metal")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework MetalKit")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework MetalPerformanceShaders")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework CoreImage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Foundation ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Cocoa")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework IOKit")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework CoreVideo")
        set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES LINK_FLAGS "-Xlinker,-rpath,@loader_path/../Frameworks")
        if(DEHANCER_USE_NATIVE_APPLE_API)
            add_definitions(-DDEHANCER_USE_NATIVE_APPLE_API=1)
        endif()
    elseif(DEHANCER_GPU_OPENCL)
        message(STATUS "Apple uses OpenCL...")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    endif()
endif()

set(BUILD_SHARED_LIBS OFF)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -EHsc -Wno-deprecated-declarations -Wno-unused-function")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    add_definitions(/MT /DWIN32=1 /D_ITERATOR_DEBUG_LEVEL=0 /D_USE_MATH_DEFINES)

    string(TOLOWER  ${CMAKE_BUILD_TYPE} BUILD_TYPE)

    if (${BUILD_TYPE} STREQUAL "debug")
        add_definitions(/MTd /D_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH=1 /D_HAS_ITERATOR_DEBUGGING=0)
    endif()

    if (DEHANCER_DEBUG)
        string(REPLACE "/O2" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
        string(REPLACE "/O2" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/Ob2" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
        string(REPLACE "/Ob2" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
        string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG=1 -DZ7=1")
        add_definitions(/DEBUG /Z7)
    endif ()

    if (PRINT_KERNELS_DEBUG)
        add_compile_definitions(PRINT_KERNELS_DEBUG=1)
    endif ()

    set(VCPKG_BUILD_TYPE release)
    set(VCPKG_LIBRARY_LINKAGE static)
    SET(NODEFAULTLIB_COMPILE_FLAGS "-Zl")
    SET(NODEFAULTLIB_LINK_FLAGS "")
    SET(VCPKG_C_FLAGS  "${VCPKG_C_FLAGS} ${NODEFAULTLIB_COMPILE_FLAGS}")
    SET(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} ${NODEFAULTLIB_LINK_FLAGS}" )

else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -Wno-unused-command-line-argument -Wno-missing-braces -Wno-unused-parameter")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Wno-missing-braces -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    if (PRINT_KERNELS_DEBUG)
        add_compile_definitions(PRINT_KERNELS_DEBUG=1)
    endif ()
endif ()

#
# Packages
#
if(WIN32)
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH};/usr/local/lib/pkgconfig")
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH};/usr/local/lib64/pkgconfig")
    set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH};$ENV{SYSROOT};/usr/local/dehancer/lib/pkgconfig")
else()
    set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig)
    set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib64/pkgconfig)
    set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:$ENV{SYSROOT}:/usr/local/dehancer/lib/pkgconfig)
endif ()


if(USE_OPENCOLORIO)
    message(STATUS "USE OPENCOLORIO")
    #pkg_check_modules(OpenColorIO OpenColorIO)
    #pkg_check_modules(yaml_cpp yaml-cpp REQUIRED)
    #find_package (yaml_cpp REQUIRED)
else()
    message(STATUS "Skip OPENCOLORIO")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    if (NOT OPENCV_FRAMEWORK_PATH)
        message(FATAL "OPENCV_FRAMEWORK_PATH variable not set")
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -F${OPENCV_FRAMEWORK_PATH} -framework opencv2")
    include_directories(${OPENCV_INCLUDES_PATH})
else()
    if (APPLE)
        find_package (OpenCV REQUIRED PATHS ${OpenCV_DIR})
    else ()
        if(WIN32)
            find_package(OpenCV REQUIRED)
            add_definitions(/DCV_IGNORE_DEBUG_BUILD_GUARD=1)
        else()
            find_package (OpenCV REQUIRED)
        endif()
    endif ()
endif()

if(LINUX)

    find_package (lapack)

    if (LAPACK_NOT_FOUND)
        message(FATAL "LAPACK has not bin found at all. $ git clone https://github.com/Reference-LAPACK/lapack-release and install")
    else()
        message("LAPACK LIBRARY: ${LAPACK_LIBRARIES}")
    endif ()

else()

    find_package (LAPACK REQUIRED)
    find_package (BLAS REQUIRED)

endif()

if (LINUX)
    set(CMAKE_INSTALL_LIBDIR  "lib")
    set(DEHANCER_EXT_MATH_LIB)

    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    set(FORTRAN_LIB_PATHS ${CMAKE_SYSTEM_LIBRARY_PATH};/usr/lib/gcc/x86_64-linux-gnu/9;/usr/lib/gcc/x86_64-linux-gnu/8;/usr/lib/gcc/x86_64-linux-gnu;/lib64;/lib;/usr/lib64;/usr/lib;/usr/local/lib64;/usr/local/lib;/opt/local/lib64;/opt/local/lib)

    find_library(GFORTRAN_LIBRARY
            NAMES gfortran
            PATHS ${FORTRAN_LIB_PATHS}
            )

    find_library(GQUADMATH_LIBRARY
            NAMES quadmath
            PATHS ${FORTRAN_LIB_PATHS}
            )

    message("GFORTRAN_LIBRARY: ${GFORTRAN_LIBRARY}")

    #    foreach(v ${BLAS_LIBRARIES})
    #        string(REPLACE ".so" ".a" vv ${v})
    #        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    #    endforeach()
    #
    #    foreach(v ${LAPACK_LIBRARIES})
    #        string(REPLACE ".so" ".a" vv ${v})
    #        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    #    endforeach()

    #    foreach(v ${GFORTRAN_LIBRARY})
    #        string(REPLACE ".so" ".a" vv ${v})
    #        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    #    endforeach()
    #
    #    foreach(v ${GQUADMATH_LIBRARY})
    #        string(REPLACE ".so" ".a" vv ${v})
    #        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    #    endforeach()

    set(DEHANCER_EXT_MATH_LIB ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    find_library(GFORTRAN_LIB NAMES gfortran)
    if (gfortran_FOUND)
        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${GFORTRAN_LIB};m)
    else()
        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};m)
    endif()

    find_library(DEHANCER_EXT_MATH_LIB REQUIRED ${DEHANCER_EXT_MATH})

else()
    set(DEHANCER_EXT_MATH_LIB ${BLAS_LIBRARIES};${LAPACK_LIBRARIES})
endif ()
include_directories(${BLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
message("BLASS/LAPACK: ${DEHANCER_EXT_MATH_LIB}")

if (DEHANCER_GPU_CUDA)

    if(WIN32)

        if (NOT CUDAToolkit_ROOT)
            set(CUDA_INSTALLATION_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")

            set(CUDA_INSTALLATION_DIR_V115 "${CUDA_INSTALLATION_DIR}/v11.5")
            set(CUDA_INSTALLATION_DIR_V116 "${CUDA_INSTALLATION_DIR}/v11.6")
            set(CUDA_INSTALLATION_DIR_V118 "${CUDA_INSTALLATION_DIR}/v11.8")
            set(CUDA_INSTALLATION_DIR_V121 "${CUDA_INSTALLATION_DIR}/v12.1")
            set(CUDA_INSTALLATION_DIR_V123 "${CUDA_INSTALLATION_DIR}/v12.3")
            set(CUDA_INSTALLATION_DIR_V10 "${CUDA_INSTALLATION_DIR}/v10.2")

            if(EXISTS ${CUDA_INSTALLATION_DIR_V123})
                set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V123})
            elseif(EXISTS ${CUDA_INSTALLATION_DIR_V121})
                set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V121})
            elseif(EXISTS ${CUDA_INSTALLATION_DIR_V118})
                    set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V118})
            elseif(EXISTS ${CUDA_INSTALLATION_DIR_V116})
                    set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V116})
            elseif(EXISTS ${CUDA_INSTALLATION_DIR_V115})
                set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V115})
            elseif(EXISTS ${CUDA_INSTALLATION_DIR_V10})
                set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V10})
            else()
                message(FATAL_ERROR "NO CUDAToolkit is found...")
            endif()

        endif ()

        set(MSVC_2019 "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\Llvm\\x64\\bin")
        set(MSVC_2022 "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\Llvm\\x64\\bin")

        if (NOT CUDA_BIN_COMPILER)
            IF(EXISTS ${MSVC_2019})
                set(CUDA_BIN_COMPILER ${MSVC_2019})
            elseif (EXISTS ${MSVC_2022})
                set(CUDA_BIN_COMPILER  ${MSVC_2022})
            else ()
                message(FATAL_ERROR "No MSVC found")
            endif ()
        endif ()
        set(CMAKE_CUDA_COMPILER_FLAGS  "-DWIN32 -DWIN64 -D_CONSOLE -D_MBCS /EHsc")

        #
        # TODO: OpenCL on WIN!
        #
        # set(DEHANCER_GPU_OPENCL OFF)

    else()

        set(CMAKE_CUDA_COMPILER_FLAGS  "-D_CONSOLE")

    endif()

    set(CMAKE_CUDA_STANDARD 14)

    find_package(CUDAToolkit REQUIRED)
    set(CUDA_FOUND ${CUDAToolkit_FOUND})

    if (CUDAToolkit_FOUND)

        set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        link_directories(${CUDAToolkit_LIBRARY_DIR})

        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})

        set(CUDA_LIB_FILE CUDA::cudart)
        set(CUDART_STATIC_LIB_FILE CUDA::cudart_static)
        set(CUDART_LIB_FILE CUDA::cuda_driver)
        set(CUDA_LIBS ${CUDA_LIB_FILE};${CUDART_STATIC_LIB_FILE};${CUDART_LIB_FILE})
        set(CUDA_LIBRARY_FLAGS  "--cudart=static")

        message("CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER}")
        message("CUDAToolkit_BIN_DIR ${CUDAToolkit_BIN_DIR}")
        message("CUDAToolkit_LIBRARY_DIR ${CUDAToolkit_LIBRARY_DIR}")
        message("CUDAToolkit_LIBRARY_ROOT ${CUDAToolkit_LIBRARY_ROOT}")
        message("CUDAToolkit_TARGET_DIR ${CUDAToolkit_TARGET_DIR}")
        message("NVCC BIN: ${CMAKE_CXX_COMPILER}")
        message("CUDA_LIBS: ${CUDA_LIBS}")

        #
        # TODO: check gcc/clang version
        #
        # clang <=9.0 for cuda 10.1
        #
        if (NOT CUDA_BIN_COMPILER)
            set(CUDA_BIN_COMPILER g++)
            #set(CUDA_BIN_COMPILER clang++)
        endif ()

        message("CUDA_BIN_COMPILER ${CUDA_BIN_COMPILER}")

    else()
        message(WARNING "CUDA Toolkit is not found...")
        set(DEHANCER_GPU_CUDA OFF)
    endif ()

endif ()


#
# GPU Platforms
#
if (DEHANCER_GPU_OPENCL)
    find_package (OpenCL REQUIRED)
    if (OpenCL_FOUND)
        set(DEHANCER_GPU_OPENCL ON)
    else()
        set(DEHANCER_GPU_OPENCL OFF)
        message("OpenCL is not found on this platform")
    endif ()
    add_definitions(-DDEHANCER_GPU_OPENCL=1)
endif ()


if (DEHANCER_GPU_OPENCL)

    if (WIN32)
        find_package(dlfcn-win32 REQUIRED)

        set(CMAKE_DL_LIBS dlfcn-win32::dl)

        include_directories(
                ${dlfcn-win32_INCLUDE_DIRS}
        )
    else()
        set(CMAKE_DL_LIBS dl)
    endif ()
endif ()

if (DEHANCER_GPU_METAL)
    if (APPLE)
        set(DEHANCER_GPU_METAL ON)
    else()
        set(DEHANCER_GPU_METAL OFF)
        message("Metal is not found on this platform")
    endif ()
endif()


if (DEHANCER_GPU_CUDA)
    if (APPLE)
        message(WARNING "Apple does not support CUDA...")
        set(DEHANCER_GPU_CUDA OFF)
    else()
        if (CUDA_FOUND)

            message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
            string(STRIP "${CMAKE_CUDA_COMPILER_FLAGS}" CMAKE_CUDA_COMPILER_FLAGS)

        else()

            message(WARNING "CUDA NOT FOUND")
            set(DEHANCER_GPU_CUDA OFF)

        endif ()

    endif()
endif ()

#
# External projects
#
include(ExternalProject)
include(FetchContent)

#
# Global setttings
#
set(VERSION_FILE "version.txt")

# determine, whether we want a static binary
SET(STATIC_LINKING FALSE CACHE BOOL "Build a static binary?")

# do we want static libraries?
# When STATIC_LINKING is TRUE, than cmake looks for libraries ending
# with .a. This is for linux only!
IF(STATIC_LINKING)
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
ENDIF(STATIC_LINKING)

# set -static, when STATIC_LINKING is TRUE and set LINK_SEARCH_END_STATIC
# to remove the additional -bdynamic from the linker line.
IF(STATIC_LINKING)
    # SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
    # SET_TARGET_PROPERTIES(surface PROPERTIES LINK_SEARCH_END_STATIC 1)
ENDIF(STATIC_LINKING)

if(PRINT_DEBUG)
    if (WIN32)
        add_definitions(/DPRINT_DEBUG=1)
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPRINT_DEBUG=1")
    endif ()
endif()

set(CMAKE_MACOSX_RPATH TRUE)

#Static start
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#Set Linker flags
#set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

#
# Generators
#
file (WRITE "${VERSION_FILE}" "${PROJECT_NAME} ${PROJECT_VERSION}")
configure_file(include/dehancer/gpulib_version.h.in ${CMAKE_BINARY_DIR}/generated/gpulib_version.h)

string(STRIP "${CMAKE_CXX_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" C_FLAGS)

#
# Google Test
#
if (BUILD_TESTING)

    find_package(GTest)

    if (NOT GTest_FOUND)

        string(STRIP "${CMAKE_C_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" C_FLAGS)
        string(STRIP "${CMAKE_CXX_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" CXX_FLAGS)

        set(CMAKE_OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}" CACHE STRING "" FORCE)

        include(FetchContent)

        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG      v1.14.0
                SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest-src"
        )

        FetchContent_GetProperties(googletest)

        if(NOT googletest_POPULATED)
            FetchContent_Populate(googletest)
            add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()

    endif ()

endif ()

if (DEHANCER_GPU_OPENCL)
    add_definitions(-DCL_TARGET_OPENCL_VERSION=120)
    #
    # OpenCL
    #

    FetchContent_Declare(
            dehancer_opencl_helper
            GIT_REPOSITORY  https://github.com/dehancer/dehancer-opencl-helper
            GIT_TAG     feature/build-fetcher-deps
            SOURCE_DIR "${CMAKE_BINARY_DIR}/dehancer_opencl_helper-src"
            BINARY_DIR "${CMAKE_BINARY_DIR}/dehancer_opencl_helper-build"
    )

    FetchContent_GetProperties(dehancer_opencl_helper)
    if(NOT dehancer_opencl_helper_POPULATED)
        FetchContent_Populate(dehancer_opencl_helper)
        add_subdirectory(${dehancer_opencl_helper_SOURCE_DIR} ${dehancer_opencl_helper_BINARY_DIR})
    endif()

    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};clHelperLib)
    set(clHelper_DIR ${dehancer_opencl_helper_SOURCE_DIR}/clHelper)
    include_directories(${clHelper_DIR})
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};clHelperLib;${OpenCL_LIBRARIES};)

endif ()


pkg_check_modules(dehancer_common_cpp dehancer-common-cpp)
pkg_check_modules(dehancer_xmp_cpp dehancer-xmp-cpp)

#
# Common
#

if (dehancer_common_cpp_FOUND AND dehancer_xmp_cpp_FOUND)

    message(STATUS "dehancer_common_cpp: ${dehancer_common_cpp_LIBRARIES}, ${dehancer_common_cpp_LIBRARY_DIRS} ${dehancer_common_cpp_INCLUDE_DIRS}")
    message(STATUS "dehancer_xmp_cpp: ${dehancer_xmp_cpp_LIBRARIES}, ${dehancer_xmp_cpp_LIBRARY_DIRS} ${dehancer_xmp_cpp_INCLUDE_DIRS}")
    include_directories(
            ${dehancer_common_cpp_INCLUDE_DIRS}
            ${dehancer_xmp_cpp_INCLUDE_DIRS}
    )
    link_directories(
            ${dehancer_xmp_cpp_LIBRARY_DIRS}
            ${dehancer_common_cpp_LIBRARY_DIRS}
    )

    set(COMMON_LIBRARIES ${dehancer_xmp_cpp_LIBRARIES};${dehancer_common_cpp_LIBRARIES};${COMMON_LIBRARIES})

else ()

    FetchContent_Declare(
            dehancer_xmp_cpp
            GIT_REPOSITORY  git@github.com:dehancer/dehancer-xmp-cpp.git
            GIT_TAG     ${GIT_TAG_dehancer_xmp_cpp}
            SOURCE_DIR "${CMAKE_BINARY_DIR}/dehancer_xmp_cpp-src"
            BINARY_DIR "${CMAKE_BINARY_DIR}/dehancer_xmp_cpp-build"
    )

    FetchContent_GetProperties(dehancer_xmp_cpp)
    if(NOT dehancer_xmp_cpp_POPULATED)
        FetchContent_Populate(dehancer_xmp_cpp)
        if (DEHANCER_INSTALL_COMMON_DEPS)
            add_subdirectory(${dehancer_xmp_cpp_SOURCE_DIR} ${dehancer_xmp_cpp_BINARY_DIR})
        else()
            add_subdirectory(${dehancer_xmp_cpp_SOURCE_DIR} ${dehancer_xmp_cpp_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif ()
    endif()

    FetchContent_GetProperties(exiv2)

    if(NOT exiv2_POPULATED)
        FetchContent_Populate(exiv2)
        if (DEHANCER_INSTALL_COMMON_DEPS)
            add_subdirectory(${exiv2_SOURCE_DIR} ${exiv2_BINARY_DIR})
        else ()
            add_subdirectory(${exiv2_SOURCE_DIR} ${exiv2_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif ()
    endif()

    include_directories(
            ${base64cpp_SOURCE_DIR}/include
            ${ed25519cpp_SOURCE_DIR}/include
            ${exiv2_SOURCE_DIR}/include
            ${dehancer_common_cpp_SOURCE_DIR}/include
            ${dehancer_common_cpp_SOURCE_DIR}/external
            ${dehancer_xmp_cpp_SOURCE_DIR}/include
            ${CMAKE_CURRENT_BINARY_DIR}
    )

    set(COMMON_LIBRARIES dehancer_xmp_cpp;dehancer_common_cpp;exiv2lib;ed25519cpp;base64cpp;${COMMON_LIBRARIES})

endif ()

#
##
## Maths
##
FetchContent_Declare(
        dehancer_maths_cpp
        GIT_REPOSITORY  https://github.com/dehancer/dehancer-maths-cpp
        GIT_TAG     ${GIT_TAG_dehancer_maths_cpp}
        SOURCE_DIR "${CMAKE_BINARY_DIR}/dehancer_maths_cpp-src"
        BINARY_DIR "${CMAKE_BINARY_DIR}/dehancer_maths_cpp-build"
)

FetchContent_GetProperties(dehancer_maths_cpp)
if(NOT dehancer_maths_cpp_POPULATED)
    FetchContent_Populate(dehancer_maths_cpp)
    add_subdirectory(${dehancer_maths_cpp_SOURCE_DIR} ${dehancer_maths_cpp_BINARY_DIR})
endif()

include_directories(
        ${dehancer_maths_cpp_SOURCE_DIR}/include
        ${armadillo_SOURCE_DIR}/include
)

set(COMMON_LIBRARIES dehancer_maths_cpp;${COMMON_LIBRARIES})

if(USE_OPENCOLORIO)
    #
    # AcademySoftwareFoundation OpenColorIO. ACES and etc...
    # https://github.com/AcademySoftwareFoundation/OpenColorIO
    #
    ExternalProject_Add(OpenColorIO
            #GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/OpenColorIO
            SOURCE_DIR ${DEHANCER_GPULIB_CPP_SOURCE}/external/OpenColorIO
            INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION}
            #PREFIX OpenColorIO
            CMAKE_ARGS
            -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
            -Wno-error=dev -Wno-dev
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DDEHANCER_TARGET_ARCH=${DEHANCER_TARGET_ARCH}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}
            -DCMAKE_BUILD_TYPE=Release #${CMAKE_BUILD_TYPE}
            -DOCIO_BUILD_TESTS=OFF
            -DOCIO_USE_SSE=${DEHANCER_USE_SSE}
            -DOCIO_BUILD_GPU_TESTS=OFF
            -DBUILD_SHARED_LIBS=OFF
            -DOCIO_BUILD_APPS=OFF
            -DOCIO_BUILD_PYTHON=OFF
            -DILMBASE_STATIC_LIBRARY=ON
            -DPYSTRING_STATIC_LIBRARY=ON
            -DYAMLCPP_STATIC_LIBRARY=ON
            -DOCIO_INSTALL_EXT_PACKAGES=ALL
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            PREFIX OpenColorIO #${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES}; OpenColorIO)
    set(COMMON_LIBRARIES   libOpenColorIO.a; ${COMMON_LIBRARIES})
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES}; libpystring.a; libyaml-cpp.a; libImath-3_1.a;)

    link_directories(
            ${CMAKE_BINARY_DIR}/OpenColorIO/src/OpenColorIO-build/ext/dist/lib
    )

endif()

include_directories(
        ${CMAKE_BINARY_DIR}/generated
        ${CMAKE_BINARY_DIR}/external/include
        ${DEHANCER_GPULIB_CPP_SOURCE}/include
        ${OpenCV_INCLUDE_DIRS}
        ./
)

if (DEHANCER_GPU_CUDA)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${CUDA_LIBS})
endif ()

if (DEHANCER_GPU_OPENCL)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${OpenCL_LIBRARY})
    include_directories(
            ${OpenCL_INCLUDE_DIRS}
    )
endif ()


message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
message("OpenCV_DIR: ${OpenCV_DIR}")
add_subdirectory(lib)

set(COMMON_LIBRARIES
        ${COMMON_LIBRARIES};
        ${DEHANCER_EXT_MATH_LIB};
        ${OpenCV_LIBRARIES};
        ${EXPAT_LIBRARIES}
        ${Iconv_LIBRARIES};
        ${ZLIB_LIBRARIES};
        ${CMAKE_DL_LIBS};
        )

message("COMMON_LIBRARIES: ${COMMON_LIBRARIES}")

if (BUILD_UTILS)
    if (NOT IOS)
        add_subdirectory(utils)
    endif ()
endif ()

if(BUILD_TESTING)

    if (WIN32)
        set(COMMON_LIBRARIES ${COMMON_LIBRARIES};Secur32.lib;Shell32.lib;Iphlpapi.lib;ws2_32;kernel32;psapi)
    endif()

    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()
