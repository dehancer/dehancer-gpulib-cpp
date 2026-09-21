function(dehancer_add_metal_shaders target source)
    find_program(DEHANCER_XCRUN NAMES xcrun REQUIRED)
    set(air "${CMAKE_CURRENT_BINARY_DIR}/${target}.air")
    set(library "${CMAKE_CURRENT_BINARY_DIR}/${target}.metallib")
    file(GLOB_RECURSE headers CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/dehancer/gpu/kernels/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../*.h"
    )
    add_custom_command(OUTPUT "${air}"
        COMMAND "${DEHANCER_XCRUN}" -sdk macosx metal
            ${METAL_FLAGS}
            -I "${CMAKE_CURRENT_SOURCE_DIR}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/.."
            -I "${PROJECT_SOURCE_DIR}/include"
            -O3 -ffast-math -Wno-unused-variable
            -c "${source}" -o "${air}"
        DEPENDS "${source}" ${headers}
        VERBATIM
    )
    add_custom_command(OUTPUT "${library}"
        COMMAND "${DEHANCER_XCRUN}" -sdk macosx metallib "${air}" -o "${library}"
        DEPENDS "${air}"
        VERBATIM
    )
    add_custom_target(${target} DEPENDS "${library}")
endfunction()

function(dehancer_add_cuda_shaders target source)
    find_program(DEHANCER_XXD NAMES xxd REQUIRED)
    set(fatbin "${CMAKE_CURRENT_BINARY_DIR}/${target}.fatbin")
    set(embedded "${CMAKE_CURRENT_BINARY_DIR}/${target}.c")
    file(GLOB_RECURSE headers CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/dehancer/gpu/kernels/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/../*.h"
    )
    set(host_compiler_args)
    if(CMAKE_CUDA_HOST_COMPILER)
        list(APPEND host_compiler_args -ccbin "${CMAKE_CUDA_HOST_COMPILER}")
    elseif(CUDA_BIN_COMPILER)
        list(APPEND host_compiler_args -ccbin "${CUDA_BIN_COMPILER}")
    endif()
    add_custom_command(OUTPUT "${fatbin}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
            ${host_compiler_args}
            --use_fast_math --extra-device-vectorization --keep-device-functions
            -Wno-deprecated-declarations
            -I "${PROJECT_SOURCE_DIR}/include"
            -I "${CMAKE_CURRENT_SOURCE_DIR}"
            -I "${CMAKE_CURRENT_SOURCE_DIR}/.."
            -DCUDA_KERNEL=1
            -fatbin "${source}" -o "${fatbin}"
        DEPENDS "${source}" ${headers}
        VERBATIM
    )
    add_custom_command(OUTPUT "${embedded}"
        COMMAND "${DEHANCER_XXD}" -i -n "${target}_fatbin" "${fatbin}" "${embedded}"
        DEPENDS "${fatbin}"
        VERBATIM
    )
    add_library(${target} STATIC "${embedded}")
endfunction()

function(dehancer_add_opencl_shaders target)
    if(NOT COMMAND COMPILE_OPENCL)
        message(FATAL_ERROR "dehancer_opencl_helper must provide COMPILE_OPENCL")
    endif()
    OPENCL_INCLUDE_DIRECTORIES(
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    COMPILE_OPENCL(TestKernels.cl)
    add_library(${target} OBJECT ${EMBEDDED_OPENCL_KERNELS} Registry.cpp)
    target_link_libraries(${target} PUBLIC
        dehancer_gpulib::dehancer_gpulib_opencl
        dehancer_opencl_helper::dehancer_opencl_helper
    )
    add_executable(${target}_checker main.cpp)
    target_link_libraries(${target}_checker PRIVATE ${target})
endfunction()
