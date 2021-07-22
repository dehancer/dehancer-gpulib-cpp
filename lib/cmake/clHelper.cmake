# // ======================================================================== # //
# // Copyright 2017 Ingo Wald                                                 # //
# //                                                                          # //
# // Licensed under the Apache License, Version 2.0 (the "License");          # //
# // you may not use this file except in compliance with the License.         # //
# // You may obtain a copy of the License at                                  # //
# //                                                                          # //
# //     http://www.apache.org/licenses/LICENSE-2.0                           # //
# //                                                                          # //
# // Unless required by applicable law or agreed to in writing, software      # //
# // distributed under the License is distributed on an "AS IS" BASIS,        # //
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # //
# // See the License for the specific language governing permissions and      # //
# // limitations under the License.                                           # //
# // ======================================================================== # //

# opencl.cmake - little helper tool to better integrate opencl files into
# a cmake project. See README.md for usage instructoins
#
# ------------------------------------------------------------------
# find basic opencl runtime components
# ------------------------------------------------------------------
FIND_PACKAGE(OpenCL REQUIRED)
IF (NOT OpenCL_INCLUDE_DIRS)
	MESSAGE(ERROR "OpenCL runtime not found")
ENDIF()
INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})

# enable C++-17
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -std=c++17")

# ------------------------------------------------------------------
# find a opencl cmd-line compiler
# ------------------------------------------------------------------

#FIND_PROGRAM(INTEL_OPENCL_COMPILER "ioc64" DOC "Intel OpenCL Compiler ('ioc64', from Intel OpenCL SDK)")
FIND_PROGRAM(CLANG_COMPILER "clang" DOC "(OpenCL-capable) clang compiler")

IF (#(NOT INTEL_OPENCL_COMPILER) AND
(NOT CLANG_COMPILER))
	#  MESSAGE("Could not find _either_ Intel or Clang OpenCL compiler.")
	MESSAGE("Could not find Clang OpenCL compiler.")
	MESSAGE("Will not be able to do command-line compilation (but runtime may still work)")
ENDIF()

FIND_PROGRAM(INTEL_OPENCL_COMPILER "ioc64" DOC "intel opencl cmd-line compiler")

IF (NOT INTEL_OPENCL_COMPILER)
	MESSAGE("Could not find intel OpenCL compiler (ioc64).")
ENDIF()

# ------------------------------------------------------------------
# list of all directories the user specified for his opencl kernels
# (using OPENCL_INCLUDE_DIRECTORIES(<dir>)
SET(CLHELPER_INCLUDE_DIRS "")
MACRO (OPENCL_INCLUDE_DIRECTORIES)
	foreach(src ${ARGN})
		set(CLHELPER_INCLUDE_DIRS ${CLHELPER_INCLUDE_DIRS} -I${src})
	endforeach()
 	message("new OPENCL_INCLUDE_DIRECTORIES ${CLHELPER_INCLUDE_DIRS}")
ENDMACRO ()

# ------------------------------------------------------------------
# list of all preprocessor definitions the user specified for his
# opencl kernels (using OPENCL_ADD_DEFINITION(<dir>)
SET(OPENCL_DEFINITIONS "")
MACRO (OPENCL_ADD_DEFINITION)
	SET(OPENCL_DEFINITIONS ${OPENCL_DEFINITIONS} ${ARGN})
	message("new OPENCL_DEFINITIONS ${OPENCL_DEFINITIONS}")
ENDMACRO ()

# list of all asm-outputs we generated for .cl files
SET(CLHELPER_ASM_FILES "")
# list of all llvm-outputs we generated for .cl files
SET(CLHELPER_LL_FILES "")
# list of all dependency files we generates for .cl files
SET(CLHELPER_DEP_FILES "")

# ------------------------------------------------------------------
# the main 'COMPILE_CL()' macro we use for compiling .cl files
#
# usage:
# - pass a list of .cl files to this macro
# - the macro will iterate over these files, and, for each
#   - generate a '.dep' file that tracks its depenencies
#   - run the c preprocessor to expand all preproc macros
#     (including the OPENCL_ADD_DEFINITIONS() and
#     OPENCL_INCLUDE_DIRECTORIES() calls made before
#     calling this macro)
#   - test-run clang in opencl mode on that expanded kernel,
#     to make sure it compiles
#   - generate, using the 'xxd' tool, a c file that
#     has the fully expanded kernel embedded as a char[]
#     array (ie, we can link the kernel source into the
#     binary)
# ------------------------------------------------------------------
MACRO (COMPILE_OPENCL)
	SET(EMBEDDED_OPENCL_KERNELS "")

	#  message("compile OPENCL_DEFINITIONS ${OPENCL_DEFINITIONS}")
	FOREACH(src ${ARGN})

		GET_FILENAME_COMPONENT(fname ${src} NAME_WE)
		GET_FILENAME_COMPONENT(abs_path ${CMAKE_CURRENT_SOURCE_DIR}/${src} PATH)
		#GET_FILENAME_COMPONENT(abs_path ${src} PATH)
		GET_FILENAME_COMPONENT(rel_path ${src} PATH)

		# the directory we're going to put all generated output files
		SET(clhelper_base_output_dir ${CMAKE_BINARY_DIR}/.clhelper)
		SET(clhelper_output_dir ${clhelper_base_output_dir}/${rel_path})

		# full path to the input file
		SET(input_file ${CMAKE_CURRENT_SOURCE_DIR}/${src})
		#SET(input_file ${abs_path})

		message("{fname} ${fname}")
		message("{abs_path} ${abs_path}")
		message("{rel_path} ${rel_path}")
		message("{input_file} ${input_file}")
		message("{clhelper_base_output_dir} ${clhelper_base_output_dir}")
		message("{clhelper_output_dir} ${clhelper_output_dir}")

		# the '.dep' file to track dependencies (generated using clang -s)
		SET(dep_file ${clhelper_output_dir}/${fname}.dep)

		# the c-preprocessor output of the input file (generated using clang -E)
		SET(preproc_file ${clhelper_output_dir}/${fname}.cl)

		# the .ll and .s files we generate using clang's opencl compiler
		SET(ll_file ${clhelper_output_dir}/${fname}.ll)
		SET(asm_file ${clhelper_output_dir}/${fname}.s)

		# the 'embedded' file that contains the (preprocessed) cl kernel
		# embedded as a global array of char[]'s
		SET(embedded_file ${clhelper_output_dir}/${fname}.embedded.c)


		IF (ALREADY_COMPILED_${src})
			# this files is already compiled ... ignore, else we get multiply defined targets
		ELSE()

			SET(ALREADY_COMPILED_${src} ON)

			# =======================================================
			# generate the dir we use to store all generated files
			# =======================================================
			ADD_CUSTOM_COMMAND(
					OUTPUT ${clhelper_output_dir}
					COMMAND ${CMAKE_COMMAND} -E make_directory ${clhelper_output_dir}
					COMMENT "Created cl-helper output directory"
			)

			# =======================================================
			# load dependencies from .dep file, IF this exists
			# =======================================================
			SET(deps "")
			IF (EXISTS ${dep_file})
				FILE(READ ${dep_file} contents)

				STRING(REPLACE " " ";"     contents "${contents}")
				STRING(REPLACE "\\" ""     contents "${contents}")
				STRING(REPLACE "\n" ";"    contents "${contents}")

				# remove first element - this is the outfile file name
				LIST(REMOVE_AT contents 0)
				FOREACH(dep ${contents})
					IF (EXISTS ${dep})
						SET(deps ${deps} ${dep})
					ENDIF (EXISTS ${dep})
				ENDFOREACH(dep ${contents})
			ENDIF()

			# ------------------------------------------------------------------
			# command to (re-)_generate_ a 'dep' file. this file is mainly
			# used for dependency tracking during build
			# ------------------------------------------------------------------
			FILE(RELATIVE_PATH rel_dep_file ${CMAKE_BINARY_DIR} ${dep_file})
			ADD_CUSTOM_COMMAND(
					OUTPUT ${dep_file}
					COMMAND ${CMAKE_COMMAND} -E make_directory ${clhelper_output_dir}
					COMMAND ${CLANG_COMPILER} -MM
					${CLHELPER_INCLUDE_DIRS}
					${CLHELPER_DEFINITIONS}
					${input_file}
					-o ${dep_file}
					DEPENDS ${input_file} ${deps}
					COMMENT "*** OPENCL ${CLANG_COMPILER} Generated dependencies for ${src} -> ${rel_dep_file}"
			)
			#      ADD_CUSTOM_TARGET(clhelper_dep_file_for_${src} ALL DEPENDS ${dep_file})

			# ------------------------------------------------------------------
			# command to generate a #include-expanded '.cl' file. this is the
			# c-preprocessor expanded opencl source file (with all #include's,
			# #defines's etc expanded. this is the 'program' that actually
			# gets embedded as a compile-time string into the executable
			# ------------------------------------------------------------------
			FILE(RELATIVE_PATH rel_preproc_file ${CMAKE_BINARY_DIR} ${preproc_file})
			ADD_CUSTOM_COMMAND(
					OUTPUT ${preproc_file}
					COMMAND ${CLANG_COMPILER} -E -DCLANG_OPENCL -DCLANG_OPENCL_PREPROC=1
					${CLHELPER_INCLUDE_DIRS}
					${CLHELPER_DEFINITIONS}
					${OPENCL_DEFINITIONS}
					${input_file}
					#-std=cl1.2
					-o ${preproc_file}
					DEPENDS ${input_file} ${deps} ${dep_file}
					COMMENT "*** OPENCL ${CLANG_COMPILER} Run pre-processor on ${src} -> ${rel_preproc_file}"
			)

			# ------------------------------------------------------------------
			# command to generate .s output file
			# ------------------------------------------------------------------
			FILE(RELATIVE_PATH rel_asm_file ${CMAKE_BINARY_DIR} ${asm_file})
			ADD_CUSTOM_COMMAND(
					OUTPUT ${asm_file}
					COMMAND ${INTEL_OPENCL_COMPILER}
					-device=cpu
					-cmd=build
					-input=${preproc_file}
					-asm=${asm_file}
					-bo="-cl-std=CL2.0"
					DEPENDS ${preproc_file}
					COMMENT "test-compiling ${rel_preproc_file} -> ${rel_asm_file}"
			)

			# ------------------------------------------------------------------
			# command to generate .ll output file
			# ------------------------------------------------------------------
			FILE(RELATIVE_PATH rel_ll_file ${CMAKE_BINARY_DIR} ${ll_file})
			ADD_CUSTOM_COMMAND(
					OUTPUT ${ll_file}
					COMMAND ${INTEL_OPENCL_COMPILER}
					-device=cpu
					-cmd=build
					-input=${preproc_file}
					-llvm=${ll_file}
					-bo="-cl-std=CL2.0"
					DEPENDS ${preproc_file}
					COMMENT "test-compiling ${rel_preproc_file} -> ${rel_ll_file}"
			)

			# ------------------------------------------------------------------
			# command to generate 'embedded' c file that contains the
			# preprocessed kernel as a string. execute that from the temp
			# subdirectory to get the name of the embedded array right - xxd
			# encodes the entire path of the input file as name of the
			# kernel
			# ------------------------------------------------------------------

			IF (INTEL_OPENCL_COMPILER)
				SET(outputs ${preproc_file} ${deps} ${asm_file} ${ll_file})
			ELSE()
				SET(outputs ${preproc_file} ${deps})
			ENDIF()
			FILE(RELATIVE_PATH rel_embedded_file ${CMAKE_BINARY_DIR} ${embedded_file})
			FILE(RELATIVE_PATH rel_input ${clhelper_base_output_dir} ${preproc_file})

			message("{embedded_file} ${embedded_file}")
			message("{rel_input} ${rel_input}")
			message("{clhelper_base_output_dir} ${clhelper_base_output_dir}")

			ADD_CUSTOM_COMMAND(
					OUTPUT ${embedded_file}
					WORKING_DIRECTORY ${clhelper_base_output_dir}
					COMMAND xxd
					-i ${rel_input}
					${embedded_file}
					DEPENDS ${outputs}
					COMMENT "embedding opencl code from ${src} -> ${rel_embedded_file}"
			)
		ENDIF()
		SET(EMBEDDED_OPENCL_KERNELS ${EMBEDDED_OPENCL_KERNELS} ${embedded_file})
	ENDFOREACH()
ENDMACRO()


