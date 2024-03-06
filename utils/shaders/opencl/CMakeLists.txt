set(CURRENT_TARGET_NAME TestKernels_opencl)
set(CURRENT_TARGET_SYNTAX_CHECKER ${CURRENT_TARGET_NAME}_checker)

INCLUDE(${clHelper_DIR}/clHelper.cmake)

file (GLOB SOURCES ${SOURCES}
        ./main.cpp
        Registry.cpp
        )

include_directories(
        ${DEHANCER_GPULIB_CPP_SOURCE}/tests/shaders
)

OPENCL_INCLUDE_DIRECTORIES(
        ${DEHANCER_GPULIB_CPP_SOURCE}/include
        ${DEHANCER_GPULIB_CPP_SOURCE}/tests/shaders
        .
)

COMPILE_OPENCL(TestKernels.cl)

message("OPEN CL Embedded sources: ${EMBEDDED_OPENCL_KERNELS}")

add_library(${CURRENT_TARGET_NAME}  ${EMBEDDED_OPENCL_KERNELS})

file (GLOB DEPENDS_OPENCL_SOURCES ${DEPENDS_OPENCL_SOURCES}
        ${DEHANCER_GPULIB_CPP_SOURCE}/include/dehancer/gpu/kernels/opencl/*.h
        ${DEHANCER_GPULIB_CPP_SOURCE}/include/dehancer/gpu/kernels/*.h
        )

add_executable(${CURRENT_TARGET_SYNTAX_CHECKER} ${SOURCES} ${DEPENDS_OPENCL_SOURCES})

set (PROJECT_LIB_PLATFORM ${PROJECT_LIB}_opencl)

target_link_libraries(
        ${CURRENT_TARGET_SYNTAX_CHECKER}
        ${CURRENT_TARGET_NAME}
        ${PROJECT_LIB_PLATFORM}
        clHelperLib
        ${COMMON_LIBRARIES}
)
