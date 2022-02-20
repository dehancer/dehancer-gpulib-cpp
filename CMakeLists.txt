cmake_minimum_required(VERSION 3.12)

#
# Project
#
set(PROJECT_LIB dehancer_gpulib)
set(DEHANCER_GPULIB_VERSION 1)
project(${PROJECT_LIB} VERSION 1.3.0)

#
# Options
#
option (BUILD_TESTING "Enable creation of  tests." OFF)
option (DEHANCER_GPU_METAL "Dehancer GPU Layer is Metal" ON)
option (DEHANCER_GPU_OPENCL "Dehancer GPU Layer is OpenCL" ON)
option (DEHANCER_GPU_CUDA "Dehancer GPU Layer is CUDA" ON)
option (DEHANCER_TARGET_ARCH "Dehancer platform arch" "")
option (VCPKG_TARGET_TRIPLET "")
option (DEHANCER_COMMON_DIR "")
option (PRINT_KERNELS_DEBUG OFF)
option (PRINT_DEBUG ON)
option (DEHANCER_DEBUG OFF)
option (USE_OPENCOLORIO OFF)

if (NOT DEHANCER_COMMON_DIR)
    set(DEHANCER_COMMON_DIR "${CMAKE_BINARY_DIR}/tmp")
endif ()
message("DEHANCER_COMMON_DIR: ${DEHANCER_COMMON_DIR}")

if (DEHANCER_TARGET_ARCH)
    #
    # x86_64-apple-macos10.13
    # or
    # arm64-apple-macos11
    #
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target ${DEHANCER_TARGET_ARCH}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target ${DEHANCER_TARGET_ARCH}")
endif ()

#
#
# C++
#
set(CMAKE_CXX_STANDARD 17)

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
find_program(MAKE_EXE NAMES  nmake make)

#
# CCache settings
#
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)


if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(LINUX)
    set(STATIC_LINKING OFF)
    message(STATUS "Linux configuration...")
    if (DEHANCER_GPU_METAL)
        message("Linux GPU Layer does not support Metal SDK")
        set(DEHANCER_GPU_METAL OFF)
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -W -Wall -Wextra -Wno-ignored-qualifiers -Wvariadic-macros")
    #set(CMAKE_CXX_FLAGS "-fvisibility=hidden -Wno-pedantic -Wunused-function")
    set(CMAKE_CXX_FLAGS "-fvisibility=default -Wno-pedantic -Wunused-function")
elseif(APPLE)
    set(STATIC_LINKING ON)
    message(STATUS "Apple configuration...")
    if (DEHANCER_GPU_METAL)
        message(STATUS "Apple uses Metal...")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ObjC++ -fvisibility=hidden -Wno-unused-command-line-argument -F/Library/Frameworks ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ObjC++ -fvisibility=default -Wno-unused-command-line-argument -F/Library/Frameworks ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ObjC++ -Wno-unused-command-line-argument -F/Library/Frameworks ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Metal")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework MetalKit")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework MetalPerformanceShaders")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework CoreImage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework AppKit")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework ServiceManagement")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Foundation ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Cocoa")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Security")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework IOKit")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework CoreVideo")
        set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES LINK_FLAGS "-Xlinker,-rpath,@loader_path/../Frameworks")
    elseif(DEHANCER_GPU_OPENCL)
        message(STATUS "Apple uses OpenCL...")
    endif()
endif()

set(BUILD_SHARED_LIBS OFF)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -EHsc -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    add_definitions(/MT /DWIN32=1 /D_ITERATOR_DEBUG_LEVEL=0)

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
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -Wno-unused-command-line-argument")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default -Wno-unused-command-line-argument")
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
    #set(ENV{OpenCV_DIR} "$ENV{OpenCV_DIR};$ENV{SYSROOT};C:/Users/denn/Dehancer/local/dehancer")
else()
    set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig)
    set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib64/pkgconfig)
    set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:$ENV{SYSROOT}:/usr/local/dehancer/lib/pkgconfig)
endif ()

find_package (Threads REQUIRED)
find_package (PkgConfig REQUIRED)
find_package (ZLIB REQUIRED)
find_package (EXPAT REQUIRED)
find_package (Iconv REQUIRED)
if(USE_OPENCOLORIO)
    message("USE OPENCOLORIO")
    #pkg_check_modules(OpenColorIO OpenColorIO)
    #pkg_check_modules(yaml_cpp yaml-cpp REQUIRED)
    #find_package (yaml_cpp REQUIRED)
else()
    message("Skip OPENCOLORIO")
endif()

if (${DEHANCER_TARGET_ARCH} MATCHES "arm64.*")
    find_package (OpenCV REQUIRED PATHS /usr/local/arm64)
elseif (${DEHANCER_TARGET_ARCH} MATCHES "x86_64.*")
    find_package (OpenCV REQUIRED PATHS /usr/local/x86_64)
else()
    if(WIN32)
        find_package(OpenCV REQUIRED)
        add_definitions(/DCV_IGNORE_DEBUG_BUILD_GUARD=1)
    else()
        find_package (OpenCV REQUIRED )
    endif()
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if (LINUX)
    set(DEHANCER_EXT_MATH_LIB)

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

    foreach(v ${BLAS_LIBRARIES})
        string(REPLACE ".so" ".a" vv ${v})
        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    endforeach()

    foreach(v ${LAPACK_LIBRARIES})
        string(REPLACE ".so" ".a" vv ${v})
        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    endforeach()

    foreach(v ${GFORTRAN_LIBRARY})
        string(REPLACE ".so" ".a" vv ${v})
        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    endforeach()

    foreach(v ${GQUADMATH_LIBRARY})
        string(REPLACE ".so" ".a" vv ${v})
        set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};${vv})
    endforeach()

    set(DEHANCER_EXT_MATH_LIB ${DEHANCER_EXT_MATH_LIB};m)

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
            set(CUDA_INSTALLATION_DIR_V10 "${CUDA_INSTALLATION_DIR}/v10.2")

            if(EXISTS ${CUDA_INSTALLATION_DIR_V116})
                set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V116})
            elseif(EXISTS ${CUDA_INSTALLATION_DIR_V115})
                set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V115})
            elseif(EXISTS ${CUDA_INSTALLATION_DIR_V10})
                set(CUDAToolkit_ROOT ${CUDA_INSTALLATION_DIR_V10})
            else()
                message(FATAL_ERROR "NO CUDAToolkit is found...")
            endif()
        endif ()

        if (NOT CUDA_BIN_COMPILER)
            set(CUDA_BIN_COMPILER "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\Llvm\\x64\\bin")
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

        #
        # TODO: check gcc/clang version
        #
        # clang <=9.0 for cuda 10.1
        #
        if (NOT CUDA_BIN_COMPILER)
            set(CUDA_BIN_COMPILER g++)
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

#
# Generators
#
file (WRITE "${VERSION_FILE}" "${PROJECT_NAME} ${PROJECT_VERSION}")
configure_file(include/dehancer/gpulib_version.h.in ${CMAKE_BINARY_DIR}/generated/gpulib_version.h)

string(STRIP "${CMAKE_CXX_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" C_FLAGS)

if (BUILD_TESTING)

    #
    # Google TEST
    #
    find_package(GTest)
    if (NOT GTEST_FOUND)

        ExternalProject_Add(googletest
                GIT_REPOSITORY https://github.com/google/googletest
                GIT_PROGRESS YES
                GIT_SHALLOW YES
                GIT_TAG release-1.11.0
                CMAKE_ARGS
                -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
                -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
                -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
                -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
                -DCMAKE_C_FLAGS=${C_FLAGS}
                -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
                -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                )
        set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};googletest)

        include_directories(
                ./
                ../include
                ${CMAKE_BINARY_DIR}/external/include
        )

        link_directories(
                ${CMAKE_BINARY_DIR}/external/lib
        )

    endif()
endif ()

if (DEHANCER_GPU_OPENCL)
    #
    # OpenCL
    #
    ExternalProject_Add(dhoph
            GIT_REPOSITORY https://github.com/dehancer/dehancer-opencl-helper
            GIT_PROGRESS YES
            GIT_SHALLOW YES
            CMAKE_ARGS
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DBUILD_TESTING=OFF
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}
            PREFIX ${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};dhoph;)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};clHelperLib;${OpenCL_LIBRARIES};)
endif ()

pkg_check_modules(dehancer_common_cpp dehancer-common-cpp)
if (dehancer_common_cpp_FOUND)

    message(STATUS "dehancer_common_cpp: ${dehancer_common_cpp_LIBRARIES}, ${dehancer_common_cpp_LIBRARY_DIRS} ${dehancer_common_cpp_INCLUDE_DIRS}")
    include_directories(
            ${dehancer_common_cpp_INCLUDE_DIRS}
    )
    link_directories(
            ${dehancer_common_cpp_LIBRARY_DIRS}
    )
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${dehancer_common_cpp_LIBRARIES})

else()
    ExternalProject_Add(dcmcpp
            GIT_REPOSITORY https://github.com/dehancer/dehancer-common-cpp
            GIT_PROGRESS YES
            GIT_SHALLOW YES
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DBUILD_TESTING=OFF
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}
            PREFIX ${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};dcmcpp;)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};dehancer_common_cpp;ed25519cpp;base64cpp)
endif()

#
# Maths
#
pkg_check_modules(dehancer_maths_cpp dehancer-maths-cpp)
if (dehancer_maths_cpp_FOUND)

    message(STATUS "maths: ${dehancer_maths_cpp_LIBRARIES}, ${dehancer_maths_cpp_LIBRARY_DIRS} ${dehancer_maths_cpp_INCLUDE_DIRS}")
    include_directories(
            ${dehancer_maths_cpp_INCLUDE_DIRS}
    )
    link_directories(
            ${dehancer_maths_cpp_LIBRARY_DIRS}
    )
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${dehancer_maths_cpp_LIBRARIES})
else()

    ExternalProject_Add(dmcpp
            GIT_REPOSITORY https://github.com/dehancer/dehancer-maths-cpp
            GIT_PROGRESS YES
            GIT_SHALLOW YES
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DBUILD_TESTING=OFF
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            PREFIX ${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};dmcpp)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};dehancer_maths_cpp;armadillo)
endif()

#
# XMP
#
pkg_check_modules(dehancer_xmp_cpp dehancer-xmp-cpp)
if (dehancer_xmp_cpp_FOUND)

    message(STATUS "xmp: ${dehancer_xmp_cpp_LIBRARIES}, ${dehancer_xmp_cpp_LIBRARY_DIRS} ${dehancer_xmp_cpp_INCLUDE_DIRS}")
    include_directories(
            ${dehancer_xmp_cpp_INCLUDE_DIRS}
    )
    link_directories(
            ${dehancer_xmp_cpp_LIBRARY_DIRS}
    )

    set(COMMON_LIBRARIES dehancer_xmp_cpp; exiv2; exiv2-xmp; ${COMMON_LIBRARIES})

else()
    ExternalProject_Add(dxmpcpp
            GIT_REPOSITORY https://github.com/dehancer/dehancer-xmp-cpp
            GIT_PROGRESS YES
            GIT_SHALLOW YES
            CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DDEHANCER_TARGET_ARCH=${DEHANCER_TARGET_ARCH}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}/x
            PREFIX ${DEHANCER_COMMON_DIR}/x
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};dxmpcpp)
    set(COMMON_LIBRARIES  dehancer_xmp_cpp;exiv2;exiv2-xmp;${COMMON_LIBRARIES})
endif()

if(USE_OPENCOLORIO)
    #
    # AcademySoftwareFoundation OpenColorIO. ACES and etc...
    # https://github.com/AcademySoftwareFoundation/OpenColorIO
    #
    ExternalProject_Add(OpenColorIO
            #GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/OpenColorIO
            SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/OpenColorIO
            INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION}
            #PREFIX OpenColorIO
            CMAKE_ARGS
            -Wno-error=dev -Wno-dev
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
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
            -DDEHANCER_TARGET_ARCH=${DEHANCER_TARGET_ARCH}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            PREFIX OpenColorIO #${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES}; OpenColorIO)
    set(COMMON_LIBRARIES   libOpenColorIO.a; ${COMMON_LIBRARIES}) #libOpenColorIOapphelpers.a;
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES}; libpystring.a; libyaml-cpp.a; libImath-3_1.a;${EXPAT_LIBRARIES};${Iconv_LIBRARIES};${ZLIB_LIBRARIES};) # libHalf-2_3_s.a)

    link_directories(
            ${CMAKE_BINARY_DIR}/OpenColorIO/src/OpenColorIO-build/ext/dist/lib
    )

endif()

set(clHelper_DIR ${CMAKE_SOURCE_DIR}/lib/cmake)

link_directories(
        ${CMAKE_BINARY_DIR}/external/lib
)

include_directories(
        ${CMAKE_BINARY_DIR}/generated
        ${CMAKE_BINARY_DIR}/external/include
        ${CMAKE_SOURCE_DIR}/include
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
        ${CMAKE_DL_LIBS};
        )

message("COMMON_LIBRARIES: ${COMMON_LIBRARIES}")

add_subdirectory(utils)

#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ignore:4099")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /ignore:4099")
#set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /ignore:4099")

if(BUILD_TESTING)

    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${EXPAT_LIBRARIES};${Iconv_LIBRARIES};${ZLIB_LIBRARIES};)

    if (WIN32)
        set(COMMON_LIBRARIES ${COMMON_LIBRARIES};Secur32.lib;Shell32.lib;Iphlpapi.lib;ws2_32;kernel32;psapi)
    endif()

    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()

