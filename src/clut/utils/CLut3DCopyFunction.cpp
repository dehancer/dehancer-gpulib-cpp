//
// Created by denn on 26.05.2021.
//

#include "dehancer/gpu/clut/utils/CLut3DCopyFunction.h"

namespace dehancer {
    
    CLut3DCopyFunction::CLut3DCopyFunction (const void *command_queue,
                                            const Texture &input,
                                            size_t lut_size,
                                            bool wait_until_completed,
                                            const std::string &library_path):
            Function(command_queue, "kernel_copy_3DLut", wait_until_completed, library_path),
            lut_size_(lut_size),
            image_size_(input->get_length()),
            channels_(input->get_channels())
    {
      auto mem = MemoryHolder::Make(get_command_queue(),image_size_);
      
      execute([this, input, &mem](CommandEncoder& encode){
          encode.set(input,0);
          encode.set(mem,1);
          encode.set((uint)lut_size_, 2);
          encode.set((uint)channels_, 3);
          return CommandEncoder::Size::From(input);
      });
      
      mem->get_contents(buffer_);
    }
    
    void CLut3DCopyFunction::foreach (std::function<void (uint, float, float, float)> block) {
     
      auto bytes = buffer_.data();
  
      for (uint ib = 0; ib < lut_size_ ; ++ib) {
        for (uint ig = 0; ig < lut_size_ ; ++ig) {
          for (uint ir = 0; ir < lut_size_ ; ++ir) {
            auto index = static_cast<int>(ir + lut_size_ * ig + lut_size_ * lut_size_ * ib) * channels_;
            auto r = *(bytes + index+0) ;
            auto g = *(bytes + index+1) ;
            auto b = *(bytes + index+2) ;
            block(index, r, g, b);
          }
        }
      }
    }
    
    size_t CLut3DCopyFunction::get_bytes_per_image () const {
      return image_size_/lut_size_;
    }
    
    size_t CLut3DCopyFunction::get_lut_size () const {
      return lut_size_;
    }
    
    size_t CLut3DCopyFunction::get_image_bytes () const {
      return image_size_;
    }
}
