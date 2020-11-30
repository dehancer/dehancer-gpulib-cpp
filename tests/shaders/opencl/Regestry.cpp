//
// Created by denn nevera on 27/11/2020.
//


#include <string>

extern char TestKernels_cl[];
extern int  TestKernels_cl_len;

namespace dehancer::device {

    extern std::string get_lib_path() {
      auto p = TestKernels_cl;
      return "TestKernels.cl";
    }

}