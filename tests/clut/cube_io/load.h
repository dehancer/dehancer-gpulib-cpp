//
// Created by denn on 24.04.2021.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "dehancer/Utils.h"
#include "tests/dotenv/dotenv_utils.h"
#include "tests/test_config.h"

void load_from_cube(const std::string& platform) {
  
  std::cout << "Test " << std::endl;
  
  try {
  
    dotenv::dotenv::instance().config();
  
    std::cout << std::endl;
  
    std::string file_path = DATA_DIR; file_path.append("/250d.cube");
  
    std::cout << "Open test: " << file_path << std::endl;
    
  
    dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
    std::string ext = dehancer::TextureIO::extension_for(type);
    float compression = 1.0f;
    
    for (auto device: dehancer::DeviceCache::Instance().get_device_list()) {
      
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
      
      dehancer::CLutCubeInput cube(command_queue);
  
      {
        std::ifstream cube_is(file_path, std::ostream::binary);
        cube_is >> cube;
      }
  
      auto transformed2d = dehancer::CLutTransform(command_queue, cube, dehancer::CLut::Type::lut_2d);
  
      std::string output_file =  "cube_to_square-"+platform+"-"+"cube"+"-";
      output_file.append(dehancer::device::get_name(device));
      output_file.append(ext);
  
      {
        std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
        os << dehancer::TextureOutput(command_queue, transformed2d.get_texture(), {
                .type = type,
                .compression = compression
        });
      }
  
      {
  
        output_file =  "cube_to_cube-"+platform+"-"+"cube"+"-";
        output_file.append(dehancer::device::get_name(device));
        output_file.append(".cube");
        
        std::ofstream os(output_file, std::ostream::binary | std::ostream::trunc);
        os << dehancer::CLutCubeOutput(command_queue, cube, (dehancer::CLutCubeOutput::Options){
                .resolution = dehancer::CLutCubeOutput::Options::small
        });
      }
    }
  }
  
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw e;
  }
  
}
