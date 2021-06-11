//
// Created by denn on 24.04.2021.
//

#pragma once

#include "dehancer/gpu/Lib.h"
#include "dehancer/MLutXmp.h"
#include "dehancer/Utils.h"
#include "tests/dotenv/dotenv_utils.h"
#include "tests/test_config.h"

#include <iostream>

void load_from_xmp(const std::string& platform) {
  
  std::cout << "Test " << std::endl;
  
  try {
  
    dotenv::dotenv::instance().config();
  
    auto pass = get_key();
  
    for (auto p: pass) std::cout<<(int)p<<",";
    std::cout<<std::endl;
  
    std::cout << std::endl;
  
    std::string film_path = DATA_DIR; film_path.append("/mlut.mlut");
    //std::string camera_path = DATA_DIR; film_path.append("/clut.clut");
    std::string cache_dir = "./cache/";
  
    dehancer::file::mkdir_p(cache_dir.c_str(),0777);
  
    std::cout << "Open test: " << film_path << std::endl;
  
    /*
     * * read properties
     * */
  
    auto xmp = dehancer::MLutXmp::Open(film_path, pass, cache_dir, true);
  
    if (!xmp) {
      std::cerr << xmp.error().message() << std::endl;
    }
  
    EXPECT_TRUE(xmp);
  
    std::cout << "               id: " << xmp->get_id() << std::endl;
  
    dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
    std::string ext = dehancer::TextureIO::extension_for(type);
    float compression = 1.0f;
    
    for (auto device: dehancer::DeviceCache::Instance().get_device_list()) {
      auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
  
      auto film = dehancer::FilmProfile(command_queue);
  
      auto error = film.load(*xmp);
  
      if (error) {
        std::cerr << error << std::endl;
      }
  
      EXPECT_TRUE(!error);
      
      std::vector<dehancer::FilmProfile::Type> types = {
              dehancer::FilmProfile::under,
              dehancer::FilmProfile::normal,
              dehancer::FilmProfile::over};
  
      for (auto lut_type: types) {
        
        std::string output_file = "out-xmp-" + platform + "-" + xmp->get_id() + "-type-"+ std::to_string(lut_type) + "-";
        output_file.append(dehancer::device::get_name(device));
        output_file.append(ext);
    
        auto lut = film.get(lut_type);
  
        EXPECT_TRUE(lut);
        
        auto out_tex = dehancer::CLutTransform(command_queue, *lut, dehancer::CLut::Type::lut_2d);
        
        {
          std::ofstream ao_bench_os(output_file, std::ostream::binary | std::ostream::trunc);
          ao_bench_os << dehancer::TextureOutput(command_queue, out_tex.get_texture(), {
                  .type = type,
                  .compression = compression
          });
        }
    
      }
  
    }
  }
  
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw e;
  }
  
}
