cmake_minimum_required(VERSION 3.12)

#
# Project
#
set(PROJECT_LIB dehancer_gpulib_cpp)
set(DEHANCER_GPULIB_VERSION 1)
project(${PROJECT_LIB} VERSION 0.0.1)

#
# C++
#
set(CMAKE_CXX_STANDARD 17)

#
# Options
#
option (BUILD_TESTING "Enable creation of  tests." OFF)
option (DEHANCER_GPU_METAL "Dehancer GPU Layer is Metal" OFF)
option (DEHANCER_GPU_OPENCL "Dehancer GPU Layer is OpenCL" OFF)

set (DEHANCER_GPU_settings 0)
if (DEHANCER_GPU_METAL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEHANCER_GPU_METAL=1")
    math(EXPR DEHANCER_GPU_settings "${DEHANCER_GPU_settings} + 1")
endif()

if(DEHANCER_GPU_OPENCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEHANCER_GPU_OPENCL=1")
    math(EXPR DEHANCER_GPU_settings "${DEHANCER_GPU_settings} + 1")
endif()

if((DEHANCER_GPU_settings GREATER 1) OR (DEHANCER_GPU_settings LESS_EQUAL 0))
    message(FATAL_ERROR "GPU Layer should be set on DEHANCER_GPU_METAL or DEHANCER_GPU_OPENCL ${DEHANCER_GPU_settings}")
endif ()

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(LINUX)
    set(STATIC_LINKING OFF)
    message(STATUS "Linux configuration...")
    if (DEHANCER_GPU_METAL)
        message(FATAL_ERROR "Linux GPU Layer does not suppor Metal SDK")
    endif ()
elseif(APPLE)
    set(STATIC_LINKING ON)
    message(STATUS "Apple configuration...")
    if (DEHANCER_GPU_METAL)
        message(STATUS "Apple uses Metal...")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ObjC++ -fvisibility=hidden -Wno-unused-command-line-argument -F/Library/Frameworks ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Metal")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework MetalKit")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework MetalPerformanceShaders")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework CoreImage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework AppKit")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework ServiceManagement")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Foundation ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Cocoa")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Security")
        set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES LINK_FLAGS "-Xlinker,-rpath,@loader_path/../Frameworks")
    elseif(DEHANCER_GPU_OPENCL)
        message(STATUS "Apple uses OpenCL...")
    endif()
endif()

#
# Global setttings
#
set(VERSION_FILE "version.txt")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -Wno-unused-command-line-argument")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dcimg_display=0 -Dcimg_use_jpeg=1 -Dcimg_use_tiff=1 -Dcimg_use_png=1")
set(BUILD_SHARED_LIBS OFF)

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
    #SET(CMAKE_EXE_LINKER_FLAGS "-static")
    #SET_TARGET_PROPERTIES(surface PROPERTIES LINK_SEARCH_END_STATIC 1)
ENDIF(STATIC_LINKING)

if(PRINT_DEBUG)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DPRINT_DEBUG=1")
endif()

set(CMAKE_MACOSX_RPATH TRUE)

# googletes, etc
set (EXTERNAL_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external)
set (EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
set (EXTERNAL_INCLUDE_DIR ${EXTERNAL_INSTALL_LOCATION}/include)
set (COMMON_DEPENDECIES ${CURRENT_METAL_TARGET_NAME})
set (COMMON_LIBRARIES)

if (NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif ()

#
# Programms
#
find_program(CCACHE_FOUND ccache)
find_program(MAKE_EXE NAMES gmake nmake make)

#
# CCache settings
#
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)


#
# Generators
#
file (WRITE "${VERSION_FILE}" "${PROJECT_NAME} ${PROJECT_VERSION}")
configure_file(include/dehancer/gpulib_version.h.in ${CMAKE_BINARY_DIR}/generated/gpulib_version.h)

#
# Packages
#
set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig)
set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib64/pkgconfig)
set(ENV{PKG_CONFIG_PATH} $ENV{SYSROOT}/usr/local/dehancer/lib/pkgconfig)
find_package (PkgConfig REQUIRED)
find_package (Threads REQUIRED)
find_package (PkgConfig REQUIRED)
find_package (BLAS REQUIRED)
find_package (LAPACK REQUIRED)
if (DEHANCER_GPU_OPENCL)
    find_package (OpenCL REQUIRED)
endif ()
#find_package (ZLIB REQUIRED)
find_package (OpenCV REQUIRED )

#
# External projects
#
include(ExternalProject)

#
# Google TEST
#
find_package(GTest)
if (NOT GTEST_FOUND)
    ExternalProject_Add(googletest
            GIT_REPOSITORY https://github.com/google/googletest
            CMAKE_ARGS
            -DCMAKE_CXX_STANDARD=${DCMAKE_CXX_STANDARD}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            )
    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};googletest)
endif()

if (DEHANCER_GPU_OPENCL)
    #
    # OpenCL
    #
    ExternalProject_Add(dehancer_opencl_helper
            GIT_REPOSITORY https://github.com/dehancer/dehancer-opencl-helper
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${DCMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            )
    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};dehancer_opencl_helper;)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};clHelperLib;${OpenCL_LIBRARIES};)
endif ()

#
# Ed25519
#
pkg_check_modules(ed25519cpp ed25519cpp)
if (ed25519cpp_FOUND)

    include_directories(
            ${ed25519cpp_INCLUDE_DIRS}
    )
    link_directories(
            ${ed25519cpp_LIBRARY_DIRS}
    )
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${ed25519cpp_LIBRARIES})

else()
    ExternalProject_Add(ed25519cpplib
            GIT_REPOSITORY https://github.com/dnevera/ed25519cpp
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            )
    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};ed25519cpplib)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};ed25519cpp)
endif()

#
# Base64
#
pkg_check_modules(base64cpp base64cpp)
if (base64cpp_FOUND)

    message(STATUS "base64cpp: ${base64cpp_LIBRARIES}, ${base64cpp_LIBRARY_DIRS} ${base64cpp_INCLUDE_DIRS}")
    include_directories(
            ${base64cpp_INCLUDE_DIRS}
    )
    link_directories(
            ${base64cpp_LIBRARY_DIRS}
    )
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${base64cpp_LIBRARIES})

else()
    ExternalProject_Add(base64cpplib
            GIT_REPOSITORY https://github.com/dnevera/base64cpp
            CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            )
    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};base64cpplib)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};base64cpp)
endif()

#
# Common
#
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
    ExternalProject_Add(dehancer_common_cpplib
            GIT_REPOSITORY https://github.com/dehancer/dehancer-common-cpp
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${DCMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            )
    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};dehancer_common_cpplib)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};dehancer_common_cpp)
endif()

#
# Math SDKs
#
pkg_check_modules(armadillo armadillo)
if (armadillo_FOUND)

    include_directories(
            ${armadillo_INCLUDE_DIRS}
    )
    link_directories(
            ${armadillo_LIBRARY_DIRS}
    )
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${armadillo_LIBRARIES})

else()

    ExternalProject_Add(armadillo-lib
            GIT_REPOSITORY https://github.com/dehancer/armadillo-code
            GIT_TAG 9.700.x
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DBUILD_TESTING=OFF
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            )

    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};armadillo-lib)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};armadillo)

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

    ExternalProject_Add(dehancer_maths_cpplib
            GIT_REPOSITORY https://github.com/dehancer/dehancer-maths-cpp
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${DCMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            )
    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};dehancer_maths_cpplib)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};dehancer_maths_cpp)
endif()

#
# AcademySoftwareFoundation OpenColorIO. ACES and etc...
# https://github.com/AcademySoftwareFoundation/OpenColorIO
#
ExternalProject_Add(OpenColorIO
        #GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/OpenColorIO
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/OpenColorIO
        INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION}
        PREFIX OpenColorIO
        CMAKE_ARGS
        -Wno-error=dev -Wno-dev
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${DCMAKE_OSX_DEPLOYMENT_TARGET}
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        -DCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DOCIO_BUILD_TESTS=OFF
        -DOCIO_BUILD_GPU_TESTS=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DOCIO_BUILD_APPS=OFF
        -DOCIO_BUILD_PYTHON=OFF
        -DILMBASE_STATIC_LIBRARY=ON
        -DPYSTRING_STATIC_LIBRARY=ON
        -DYAMLCPP_STATIC_LIBRARY=ON
        -DOCIO_INSTALL_EXT_PACKAGES=ALL
        )
set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES}; OpenColorIO)
set(COMMON_LIBRARIES  ${COMMON_LIBRARIES}; libOpenColorIO.a; libOpenColorIOapphelpers.a)

set(clHelper_DIR ${CMAKE_SOURCE_DIR}/lib/cmake)

link_directories(
        ${CMAKE_BINARY_DIR}/external/lib
)

include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${CMAKE_BINARY_DIR}/external/include
        ${CMAKE_BINARY_DIR}/external/include/openfx
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external/OpenColorIO/src/OpenColorIO
        ${OpenCV_INCLUDE_DIRS}
        ./
)

set(COMMON_LIBRARIES
        ${COMMON_LIBRARIES};
        ${BLAS_LIBRARIES};
        ${LAPACK_LIBRARIES};
        ${OpenCV_LIBS};
        )
message("COMMON_LIBRARIES: ${COMMON_LIBRARIES}")
add_subdirectory(lib)

if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()