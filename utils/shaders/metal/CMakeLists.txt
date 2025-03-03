 set(CURRENT_TARGET_NAME "${CURRENT_KERNEL_TARGET_NAME}_metal")
 set(CURRENT_TARGET_LIB ${CURRENT_TARGET_NAME}.metallib)

 include_directories(
         ${DEHANCER_GPULIB_CPP_SOURCE}/utils/shaders
         .
 )

 set(METAL_KERNELS_LIBRARY ${CMAKE_BINARY_DIR}/utils/shaders/metal/${CURRENT_TARGET_LIB})

 configure_file(${CMAKE_CURRENT_SOURCE_DIR}/paths_config.h.in ${CMAKE_BINARY_DIR}/generated/utils/metal/paths_config.h)

 file (GLOB DEPENDS_SOURCES ${DEPENDS_PLUGIN_SOURCES}
         ${CMAKE_CURRENT_SOURCE_DIR}/*.h
         ${DEHANCER_GPULIB_CPP_SOURCE}/utils/shaders/*.h
         ${DEHANCER_GPULIB_CPP_SOURCE}/include/dehancer/gpu/kernels/metal/*.h
         ${DEHANCER_GPULIB_CPP_SOURCE}/include/dehancer/gpu/kernels/*.h
         )

 set(METAL_TARGET_FLAGS)
 if (${DEHANCER_TARGET_ARCH} MATCHES "arm64.*")
  #
  # arm64-apple-macos11
  #
  set(METAL_TARGET_FLAGS "air64-apple-macos11.0")
 else()
  #
  # x86_64-apple-macos10.13
  #
  set(METAL_TARGET_FLAGS "air64-apple-macos10.14")
 endif ()

 string(STRIP METAL_TARGET_FLAGS ${METAL_TARGET_FLAGS})

 add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_KERNEL_TARGET_NAME}.air
         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}.metal ${DEPENDS_SOURCES} ${COMMON_DEPENDENCIES}
         COMMAND xcrun -v -sdk macosx metal
         ${METAL_FLAGS}
         -I .
         -I ${CMAKE_CURRENT_SOURCE_DIR}
         -I ${DEHANCER_GPULIB_CPP_SOURCE}/include
         -I ${DEHANCER_GPULIB_CPP_SOURCE}/utils/shaders
         -O3 -ffast-math
         -c
         -target ${METAL_TARGET_FLAGS}
         ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}.metal
         -o ${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_KERNEL_TARGET_NAME}.air
         VERBATIM
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         )

 add_custom_target(
         ${CURRENT_TARGET_NAME}
         DEPENDS ${DEPENDS_SOURCES}
         ${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_KERNEL_TARGET_NAME}.air
         COMMAND xcrun -sdk macosx metallib
         ${CMAKE_CURRENT_BINARY_DIR}/*.air
         -o ${CURRENT_TARGET_LIB}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
 )

 set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES BUNDLE TRUE)
 set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES PREFIX "")
 set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES SUFFIX ".metallib")

