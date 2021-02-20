 set(CURRENT_TARGET_NAME "${CURRENT_KERNEL_TARGET_NAME}_metal")
 set(CURRENT_TARGET_LIB ${CURRENT_TARGET_NAME}.metallib)

 include_directories(
         ${CMAKE_SOURCE_DIR}/tests/shaders
         .
 )

 file (GLOB DEPENDS_SOURCES ${DEPENDS_PLUGIN_SOURCES}
         ${CMAKE_CURRENT_SOURCE_DIR}/*.h
         ${CMAKE_SOURCE_DIR}/tests/shaders/*.h
         )

 add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CURRENT_KERNEL_TARGET_NAME}.air
         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}.metal ${DEPENDS_SOURCES} ${COMMON_DEPENDENCIES}
         COMMAND xcrun -v -sdk macosx metal
         ${METAL_FLAGS}
         -I ${CMAKE_CURRENT_SOURCE_DIR}
         -I ${CMAKE_SOURCE_DIR}/include
         -I ${CMAKE_SOURCE_DIR}/tests/shaders
         -c ${CMAKE_CURRENT_SOURCE_DIR}/${CURRENT_KERNEL_TARGET_NAME}.metal
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

