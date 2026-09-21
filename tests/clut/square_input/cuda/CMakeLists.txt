if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.env)
    file (WRITE ".env" "CMLUT_KEY=")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.env)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/.env ${CMAKE_CURRENT_BINARY_DIR}/.env COPYONLY)
endif()

dehancer_add_test(cuda-square-input-test)
