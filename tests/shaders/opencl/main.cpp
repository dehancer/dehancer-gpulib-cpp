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
    auto function = dehancer::Function(q, "kernel_test_transform");
    auto function1 = dehancer::Function(q, "kernel_vec_add");
    auto function2 = dehancer::Function(q, "kernel_histogram_image");
    auto function3 = dehancer::Function(q, "kernel_sum_partial_histogram_image");
  
  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  return 0;
}

