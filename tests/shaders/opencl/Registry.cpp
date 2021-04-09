//
// Created by denn nevera on 27/11/2020.
//


#include <string>

#include "dehancer/gpu/Paths.h"

extern "C" char TestKernels_cl[];
extern  "C" int  TestKernels_cl_len;

namespace dehancer::device {

    extern std::string get_lib_path() {
      return "";
    }

    extern std::size_t get_lib_source(std::string& source) {
      source.clear();
      source.append(TestKernels_cl,TestKernels_cl_len);
      return std::hash<std::string>{}(source);
    }

}