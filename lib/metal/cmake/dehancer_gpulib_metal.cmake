add_library(${PROJECT_LIB} STATIC IMPORTED)

find_library(${PROJECT_LIB}_LIBRARY_PATH ${PROJECT_LIB} HINTS "../../..")
set_target_properties(${PROJECT_LIB} PROPERTIES IMPORTED_LOCATION "${dehancer_gpulib_metal_LIBRARY_PATH}")

include_directories(
        "${dehancer_gpulib_metal_INCLUDE_PATH}"
)