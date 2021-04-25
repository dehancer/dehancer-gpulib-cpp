//
// Created by denn on 24.04.2021.
//

#pragma once

#include "dehancer/gpu/Lib.h"

void clut_transform(const std::string& platform) {
  
  std::cout << "Test " << std::endl;
  
  try {
  
    dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
    std::string ext = dehancer::TextureIO::extention_for(type);
    float compression = 1.0f;
    
    for (auto device: dehancer::DeviceCache::Instance().get_device_list()) {
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
  
      auto clut_2d_identity_64 = dehancer::CLut2DIdentity(command_queue, 64);
      
      auto transformed3d = dehancer::CLutTransform(command_queue, clut_2d_identity_64, dehancer::CLut::Type::lut_3d);
      auto transformed2d = dehancer::CLutTransform(command_queue, transformed3d, dehancer::CLut::Type::lut_2d);
  
      std::string output_file =  "clut2d-"+platform+"-";
      output_file.append(dehancer::device::get_name(device));
      output_file.append(ext);
  
      {
        std::ofstream ao_bench_os(output_file, std::ostream::binary | std::ostream::trunc);
        ao_bench_os << dehancer::TextureOutput(command_queue, transformed2d.get_texture(), {
                .type = type,
                .compression = compression
        });
      }
      
    }
  }
  
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw e;
  }
  
}
