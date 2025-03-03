//
// Created by denn on 24.04.2021.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include <regex>

void make_identity(const std::string& platform) {
  
  std::cout << "Test " << std::endl;
  
  try {
  
    dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
    std::string ext = dehancer::TextureIO::extension_for(type);
    float compression = 1.0f;
    
    std::cout << "CLut identity extension: " << ext << std::endl;
    
    for (auto device: dehancer::DeviceCache::Instance().get_device_list()) {
  
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
      
      std::string dev_name =  std::regex_replace(dehancer::device::get_name(device), std::regex("[:., ]+"), "-");
  
      {
        /**
         * 2D
         */
        auto clut_2d_identity = dehancer::CLut2DIdentity(command_queue, 64);
    
        std::string output_file = "clut-2d-" + platform + "-";
        output_file.append(dev_name);
        output_file.append(ext);
    
        std::cout << "clut_2d_identity name: " << output_file << std::endl;
        
        {
          std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
          
          os << dehancer::TextureOutput(command_queue, clut_2d_identity.get_texture(), {
                  .type = type,
                  .compression = compression
          });
        }
      }
  
      
      {
        /**
         * HALD
         */
        auto clut_hald_identity = dehancer::CLutHaldIdentity(command_queue, 64);
    
        std::string output_file = "clut-hald-" + platform + "-";
        output_file.append(dev_name);
        output_file.append(ext);
  
        std::cout << "clut_hald_identity name: " << output_file << std::endl;
  
        {
          std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
          os << dehancer::TextureOutput(command_queue, clut_hald_identity.get_texture(), {
                  .type = type,
                  .compression = compression
          });
        }
  
        auto cube_output = dehancer::CLutCubeOutput(
                command_queue,
                clut_hald_identity,
                (dehancer::CLutCubeOutput::Options){
                        .resolution = dehancer::CLutCubeOutput::Options::Resolution::large
                });
  
        {
          output_file = "clut-hald-to-cube-" + platform + "-";
          output_file.append(dev_name);
          output_file.append(".cube");
  
          std::cout << "cube_output name: " << output_file << std::endl;
  
          std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
          os << cube_output;
        }
      }
      
    }
  }
  
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw e;
  }
  
}
