function(dehancer_embed_overlays output_variable)
    find_program(DEHANCER_XXD NAMES xxd REQUIRED)

    file(GLOB images CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/rc/watermarks/*.png"
        "${PROJECT_SOURCE_DIR}/src/rc/false_color/*.png"
    )

    set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded")
    set(sources)

    foreach(image IN LISTS images)
        get_filename_component(name "${image}" NAME_WE)
        set(symbol "dehancer_${name}")
        set(output "${output_dir}/${symbol}.c")
        add_custom_command(OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
            COMMAND "${DEHANCER_XXD}" -i -n "${symbol}" "${image}" "${output}"
            DEPENDS "${image}"
            VERBATIM
        )
        list(APPEND sources "${output}")
    endforeach()

    set(${output_variable} "${sources}" PARENT_SCOPE)
endfunction()
