cmake_minimum_required(VERSION 3.12)

#
# Project
#
set(PROJECT_LIB dehancer_gpulib)
set(DEHANCER_GPULIB_VERSION 1)
project(${PROJECT_LIB} VERSION 0.5.1)

#
# Options
#
option (BUILD_TESTING "Enable creation of  tests." OFF)
option (DEHANCER_GPU_METAL "Dehancer GPU Layer is Metal" ON)
option (DEHANCER_GPU_OPENCL "Dehancer GPU Layer is OpenCL" ON)
option (DEHANCER_GPU_CUDA "Dehancer GPU Layer is CUDA" ON)
option (DEHANCER_TARGET_ARCH "Dehancer platform arch" "")
option(VCPKG_TARGET_TRIPLET "")

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

# googletes, etc
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
find_program(MAKE_EXE NAMES gmake nmake make)

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
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -Wno-pedantic -Wunused-function -Wunused-command-line-argument")
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

set(BUILD_SHARED_LIBS OFF)

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -EHsc -Wno-deprecated-declarations")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    add_definitions(/MT /DWIN32=1 /D_ITERATOR_DEBUG_LEVEL=0)
    string(TOLOWER  ${CMAKE_BUILD_TYPE} BUILD_TYPE)
    if (${BUILD_TYPE} STREQUAL "debug")
        add_definitions(/MTd)
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -rdynamic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -Wno-unused-command-line-argument")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
endif ()

#
# Packages
#
set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig)
set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/usr/local/lib64/pkgconfig)
set(ENV{PKG_CONFIG_PATH} $ENV{SYSROOT}/usr/local/dehancer/lib/pkgconfig)

find_package (Threads REQUIRED)
find_package (PkgConfig REQUIRED)
find_package (BLAS REQUIRED)
find_package (LAPACK REQUIRED)
find_package (ZLIB REQUIRED)

if (${DEHANCER_TARGET_ARCH} MATCHES "arm64.*")
    find_package (OpenCV REQUIRED PATHS /usr/local/arm64)
elseif (${DEHANCER_TARGET_ARCH} MATCHES "x86_64.*")
    find_package (OpenCV REQUIRED PATHS /usr/local/x86_64)
else()
    find_package (OpenCV REQUIRED )
endif()

if (DEHANCER_GPU_CUDA)

    if(WIN32)

        if (NOT CUDAToolkit_ROOT)
            set(CUDAToolkit_ROOT "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2")
        endif ()

        if (NOT CUDA_BIN_COMPILER)
            set(CUDA_BIN_COMPILER "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\Llvm\\x64\\bin")
        endif ()
        set(CMAKE_CUDA_COMPILER_FLAGS  "-DWIN32 -DWIN64 -D_CONSOLE -D_MBCS /EHsc")

        #
        # TODO: OpenCL on WIN!
        #
        set(DEHANCER_GPU_OPENCL OFF)

    else()

        set(CMAKE_CUDA_COMPILER_FLAGS  "-D_CONSOLE")

    endif()

    set(CMAKE_CUDA_STANDARD 14)

    find_package(CUDAToolkit REQUIRED)

    set(CUDA_FOUND ${CUDAToolkit_FOUND})
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
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DPRINT_DEBUG=1")
endif()

set(CMAKE_MACOSX_RPATH TRUE)

#
# Generators
#
file (WRITE "${VERSION_FILE}" "${PROJECT_NAME} ${PROJECT_VERSION}")
configure_file(include/dehancer/gpulib_version.h.in ${CMAKE_BINARY_DIR}/generated/gpulib_version.h)

string(STRIP "${CMAKE_CXX_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} -D_ITERATOR_DEBUG_LEVEL=0" C_FLAGS)

#
# Google TEST
#
find_package(GTest)
if (NOT GTEST_FOUND)

    ExternalProject_Add(googletest
            GIT_REPOSITORY https://github.com/google/googletest
            CMAKE_ARGS
            -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
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

if (DEHANCER_GPU_OPENCL)
    #
    # OpenCL
    #
    ExternalProject_Add(dehancer_opencl_helper
            GIT_REPOSITORY https://github.com/dehancer/dehancer-opencl-helper
            CMAKE_ARGS
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DBUILD_TESTING=OFF
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            PREFIX ${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};dehancer_opencl_helper;)
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
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DBUILD_TESTING=OFF
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DVCPKG_TARGET_TRIPLET=x64-windows-static
            PREFIX ${DEHANCER_COMMON_DIR}
            BUILD_ALWAYS OFF
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
            CMAKE_ARGS
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
            -DBUILD_TESTING=OFF
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DVCPKG_TARGET_TRIPLET=x64-windows-static
            PREFIX ${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};dmcpp)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES};dehancer_maths_cpp;armadillo)
endif()

if(DEHANCER_USE_OPENCOLOR)
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
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
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
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            PREFIX ${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES}; OpenColorIO)
    set(COMMON_LIBRARIES  ${COMMON_LIBRARIES}; libOpenColorIO.a; libOpenColorIOapphelpers.a)
endif()

set(clHelper_DIR ${CMAKE_SOURCE_DIR}/lib/cmake)

link_directories(
        ${CMAKE_BINARY_DIR}/external/lib
)

include_directories(
        ${CMAKE_BINARY_DIR}/generated
        ${CMAKE_BINARY_DIR}/external/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external/OpenColorIO/src/OpenColorIO
        ${OpenCV_INCLUDE_DIRS}
        ./
)

if (DEHANCER_GPU_CUDA)
    set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${CUDA_LIBS})
endif ()

set(COMMON_LIBRARIES
        ${COMMON_LIBRARIES};
        ${BLAS_LIBRARIES};
        ${LAPACK_LIBRARIES};
        ${OpenCV_LIBS};
        )

message("COMMON_LIBRARIES: ${COMMON_LIBRARIES}")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
message("OpenCV_DIR: ${OpenCV_DIR}")
add_subdirectory(lib)

if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()