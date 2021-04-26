set (CURRENT_KERNEL_TARGET_NAME TestKernels)

set(IMAGES_DIR ${CMAKE_SOURCE_DIR}/tests/images)
set(DATA_DIR ${CMAKE_SOURCE_DIR}/tests/data)
configure_file(${CMAKE_SOURCE_DIR}/tests/include/test_config.h.in ${CMAKE_BINARY_DIR}/generated/tests/test_config.h)

if (DEHANCER_GPU_CUDA)
    add_subdirectory(cuda)
endif ()

find_package (EXPAT REQUIRED)
find_package (Iconv REQUIRED)
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

    set(COMMON_LIBRARIES ${COMMON_LIBRARIES}; dehancer_xmp_cpp exiv2 exiv2-xmp)

else()
    ExternalProject_Add(dxmpcpp
            GIT_REPOSITORY https://github.com/dehancer/dehancer-xmp-cpp
            CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_FLAGS=${C_FLAGS}
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
            -DDEHANCER_TARGET_ARCH=${DEHANCER_TARGET_ARCH}
            -DDEHANCER_COMMON_DIR=${DEHANCER_COMMON_DIR}
            -DEXTERNAL_INSTALL_LOCATION=${EXTERNAL_INSTALL_LOCATION}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DVCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}
            PREFIX ${DEHANCER_COMMON_DIR}
            )
    set(COMMON_DEPENDENCIES ${COMMON_DEPENDENCIES};dxmpcpp)
    set(COMMON_LIBRARIES  dehancer_xmp_cpp;exiv2;exiv2-xmp;${COMMON_LIBRARIES})
endif()

set(COMMON_LIBRARIES ${COMMON_LIBRARIES};${EXPAT_LIBRARIES};${Iconv_LIBRARIES};)


add_subdirectory(shaders)
add_subdirectory(aobench)
add_subdirectory(device)
add_subdirectory(texture)
add_subdirectory(function)
add_subdirectory(memory)
add_subdirectory(operations)
add_subdirectory(filter)
add_subdirectory(overlays)
add_subdirectory(clut)