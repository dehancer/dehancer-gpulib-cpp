//
// Created by denn nevera on 19/11/2020.
//

void dehancer_opencl_kernel_checker() {};

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/DeviceCache.h"
#include <string>

int main() {

  try {
    auto q = dehancer::DeviceCache::Instance().get_default_command_queue();
    auto function = dehancer::Function(q, "kernel_dehancer_pass");
  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}


extern char TestKernels_cl[];
extern int  TestKernels_cl_len;

namespace dehancer::device {
    extern std::string get_lib_path() {
      auto p = TestKernels_cl;
      return "TestKernels.cl";
    }
}