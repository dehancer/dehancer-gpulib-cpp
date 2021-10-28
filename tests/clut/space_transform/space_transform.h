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
    
//    auto space = (dehancer::StreamSpace) {
//            .type = dehancer::DHCR_ColorSpace,
//            .expandable = false,
//            .transform_func = transform_function,
//            .transform_lut = {
//                    .is_identity = false,
//                    .forward = dehancer::ocio::ACEScct::forward::lut::params,
//                    .inverse = dehancer::ocio::ACEScct::inverse::lut::params
//            },
//            .id = "aces_cct_ap1",
//            .name="ACEScct (AP1)",
//    };

//    auto space = (dehancer::StreamSpace) {
//            .type = dehancer::DHCR_ColorSpace,
//            .expandable = false,
//            .transform_func = transform_function,
//            .transform_lut = {
//                    .is_identity = false,
//                    .forward = dehancer::ocio::DVRWGIntermediate::forward::lut::params,
//                    .inverse = dehancer::ocio::DVRWGIntermediate::inverse::lut::params
//            },
//            .id = "dvr_wg_intermediate",
//            .name="DVR WG/Intermediate",
//    };
  
    auto space = (dehancer::StreamSpace) {
            .type = dehancer::DHCR_ColorSpace,
            .expandable = false,
            .transform_func = transform_function,
            .transform_lut = {
                    .is_identity = false,
                    .forward = dehancer::ocio::CineonLog::forward::lut::params,
                    .inverse = dehancer::ocio::CineonLog::inverse::lut::params
            },
            .id = "cineon_film_log",
            .name="Cineon Film Log",
    };
  
    for (auto device: dehancer::DeviceCache::Instance().get_device_list()) {
      
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
      
      auto clut_2d_identity = dehancer::CLut2DIdentity(command_queue, 64);
      
      auto output =  dehancer::TextureOutput(command_queue, clut_2d_identity.get_texture(), {
              .type = type,
              .compression = compression
      });
  
      std::string dev_name =  std::regex_replace(dehancer::device::get_name(device), std::regex("[:., ]+"), "-");
  
      std::string identity_file =  "space-identity-"+platform+"-"+"forward"+"-";
      identity_file.append(dev_name);
      identity_file.append(ext);
  
      {
        std::ofstream os(identity_file, std::ostream::binary | std::ostream::trunc);
        os << output;
      }
      
      auto transformer = dehancer::StreamTransform(command_queue,
                                                   nullptr,
                                                   nullptr,
                                                   space,
                                                   DHCR_Inverse);
      
      transformer.set_impact(1.0f);
      transformer.set_source(clut_2d_identity.get_texture());
      transformer.set_destination(output.get_texture());
      
      transformer.process();
      
      std::string output_file =  "space-transform-"+platform+"-"+"inverse"+"-";
      output_file.append(dev_name);
      output_file.append(ext);
      
      {
        std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
        os << output;
      }
      
      transformer.set_direction(DHCR_Forward);
      transformer.set_source(output.get_texture());
      
      transformer.process();
      
      output_file =  "space-transform-"+platform+"-"+"forward"+"-";
      output_file.append(dev_name);
      output_file.append(ext);
      
      {
        std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
        os << output;
      }
      
    }
  }
  
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw e;
  }
  
}
