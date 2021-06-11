//
// Created by denn nevera on 2019-07-23.
//

#include "dehancer/gpu/clut/CLutHaldIdentity.h"

namespace dehancer {
    
    CLutHaldIdentity::CLutHaldIdentity(
            const void *command_queue,
            uint lut_size,
            bool wait_until_completed
            ):
            Command(command_queue, wait_until_completed),
            CLut(),
            level_((size_t)std::sqrt((float)lut_size)),
            lut_size_(lut_size)
    {
      
      std::vector<float> buffer; buffer.resize(lut_size_*lut_size_*lut_size_*4);
      auto p = (float*)buffer.data();
      auto denom = float(lut_size_ - 1);
      
      for(size_t i = 0; i < lut_size_; i++)
      {
        for(size_t j = 0; j < lut_size_; j++)
        {
          for(size_t k = 0; k < lut_size_; k++)
          {
            *p = (float)k / denom; ++p;
            *p = (float)j / denom; ++p;
            *p = (float)i / denom; ++p;
            *p = 1.0f; ++p;
          }
        }
      }
  
      auto size = level_*level_*level_;
      
      TextureDesc desc = {
              .width = size,
              .height = size,
              .depth = 1,
              .pixel_format = TextureDesc::PixelFormat::rgba32float,
              .type = TextureDesc::Type::i2d,
              .mem_flags = TextureDesc::MemFlags::read_write
      };
  
      texture_ = desc.make(get_command_queue(), buffer.data());
      
    }
  
}