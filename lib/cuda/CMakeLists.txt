
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEHANCER_GPU_CUDA=1")

set (PROJECT_LIB_PLATFORM ${PROJECT_LIB}_cuda)

file (GLOB COMMON_IMPL_SRC ${COMMON_IMPL_SRC}
        ${CMAKE_SOURCE_DIR}/src/platforms/cuda/*.cpp
        )

add_library(${PROJECT_LIB_PLATFORM} ${COMMON_SRC} ${COMMON_IMPL_SRC})

if (COMMON_DEPENDENCIES)
    add_dependencies(${PROJECT_LIB_PLATFORM} ${COMMON_DEPENDENCIES})
endif()


set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_LIB_PLATFORM}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_LIB_PLATFORM}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_LIB_PLATFORM}")
set(namespace "${PROJECT_LIB_PLATFORM}::")

write_basic_package_version_file(
        "${version_config}" COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/lib/cmake/Config.cmake.in"
        "${project_config}"
        INSTALL_DESTINATION "${config_install_dir}"
)

install(TARGETS ${PROJECT_LIB_PLATFORM}   DESTINATION lib)
install(FILES   ${project_config} ${version_config} DESTINATION lib/cmake/${PROJECT_LIB_PLATFORM})
install(FILES cmake/${PROJECT_LIB_PLATFORM}.cmake DESTINATION lib/cmake/${PROJECT_LIB_PLATFORM})
