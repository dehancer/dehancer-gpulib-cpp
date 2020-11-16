 set(CURRENT_TARGET_NAME "TestKernels")
 set(CURRENT_TARGET_LIB ${CURRENT_TARGET_NAME}.metallib)
 set(CMAKE_INSTALL_PREFIX "/Library/OFX/Plugins")

 include_directories(
         ${CMAKE_SOURCE_DIR}/tests/shaders
         .
 )

 file (GLOB DEPENDS_SOURCES ${DEPENDS_PLUGIN_SOURCES}
         ${CMAKE_CURRENT_SOURCE_DIR}/*.h
         ${CMAKE_SOURCE_DIR}/tests/shaders/*.h
         )

 add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TestKernels.air
         DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/TestKernels.metal ${DEPENDS_SOURCES}
         COMMAND xcrun -v -sdk macosx metal
         ${METAL_FLAGS}
         -I ${CMAKE_CURRENT_SOURCE_DIR}
         -I ${CMAKE_SOURCE_DIR}/tests/shaders
         -c ${CMAKE_CURRENT_SOURCE_DIR}/TestKernels.metal
         -o ${CMAKE_CURRENT_BINARY_DIR}/TestKernels.air
         VERBATIM
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         )

 add_custom_target(
         ${CURRENT_TARGET_NAME}
         DEPENDS
         ${CMAKE_CURRENT_BINARY_DIR}/TestKernels.air
         COMMAND xcrun -sdk macosx metallib
         ${CMAKE_CURRENT_BINARY_DIR}/*.air
         -o ${CURRENT_TARGET_LIB}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
 )

 set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES BUNDLE TRUE)
 set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES PREFIX "")
 set_target_properties(${CURRENT_TARGET_NAME} PROPERTIES SUFFIX ".metallib")