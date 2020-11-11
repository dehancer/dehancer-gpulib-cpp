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
option(BUILD_TESTING "Enable creation of  tests." OFF)

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

set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

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
# Generators
#
file (WRITE "${VERSION_FILE}" "${PROJECT_NAME} ${PROJECT_VERSION}")
configure_file(include/dehancer/gpu/gpulib_version.h.in ${CMAKE_BINARY_DIR}/generated/gpulib_version.h)

#
# Packages
#
find_package (PkgConfig QUIET)
find_package (Threads REQUIRED)
find_package (PkgConfig REQUIRED)
find_package (BLAS REQUIRED)
find_package (LAPACK REQUIRED)
find_package (OpenCL REQUIRED)
find_package (JPEG REQUIRED)
find_package (TIFF REQUIRED)
find_package (PNG REQUIRED)
find_package (ZLIB REQUIRED)
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
            )
    set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};googletest)
endif()

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

set(COMMON_DEPENDECIES ${COMMON_DEPENDECIES};dehancer_opencl_helper)
set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};clHelperLib)

set(clHelper_DIR ${CMAKE_SOURCE_DIR}/lib/cmake)

link_directories(
        ${CMAKE_BINARY_DIR}/external/lib
)

include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${CMAKE_BINARY_DIR}/external/include
        ${CMAKE_BINARY_DIR}/external/include/openfx
        ${CMAKE_SOURCE_DIR}/include
        ./
        ${OpenCV_INCLUDE_DIRS}
)

set(COMMON_LIBRARIES
        ${COMMON_LIBRARIES};
        ${BLAS_LIBRARIES};
        ${LAPACK_LIBRARIES};
        ${OpenCL_LIBRARIES};
        ${PNG_LIBRARIES};
        ${JPEG_LIBRARIES};
        ${TIFF_LIBRARIES};
        ${OpenCV_LIBS};
        )

add_subdirectory(lib)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()