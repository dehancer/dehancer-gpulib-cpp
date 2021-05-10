//
// Created by denn on 24.04.2021.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "dehancer/Utils.h"
#include "tests/dotenv/dotenv_utils.h"
#include "tests/test_config.h"

void load_from_cache(const std::string& platform) {
  
  std::cout << "Test " << std::endl;
  
  try {
    
    dotenv::dotenv::instance().config();
    
    std::cout << std::endl;
    
    dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
    std::string ext = dehancer::TextureIO::extension_for(type);
    float compression = 1.0f;
    
    dehancer::DHCR_StreamSpace_TransformFunc transform_function = dehancer::stream_space_transform_func_identity();
    
    
    auto space = (dehancer::StreamSpace) {
            .type = dehancer::DHCR_ColorSpace,
            .expandable = false,
            .transform_func = transform_function,
            .transform_lut = {
                    .is_identity = false,
                    .forward = dehancer::ocio::DEH2020::forward::lut::params,
                    .inverse = dehancer::ocio::DEH2020::inverse::lut::params
            },
            .id = "aces_cct_ap1",
            .name="ACEScct (AP1)",
    };
    
    for (auto device: dehancer::DeviceCache::Instance().get_device_list()) {
      
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
      
      auto lut = dehancer::StreamSpaceCache::Instance().get_lut(command_queue, space, DHCR_Forward);
  
      GTEST_EXPECT_TRUE(lut!= nullptr);
      
      auto transformed2d = dehancer::CLutTransform(command_queue, *lut, dehancer::CLut::Type::lut_2d);
      
      std::string output_file =  "cached_aces_to_square-"+platform+"-"+"cube"+"-";
      output_file.append(dehancer::device::get_name(device));
      output_file.append(ext);
      
      {
        std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
        os << dehancer::TextureOutput(command_queue, transformed2d.get_texture(), {
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
