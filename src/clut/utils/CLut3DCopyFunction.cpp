//
// Created by denn on 26.05.2021.
//

#include "dehancer/gpu/clut/utils/CLut3DCopyFunction.h"

namespace dehancer {

    unsigned int as_uint(const float x) {
      return *(uint*)&x;
    }
    float as_float(const unsigned int x) {
      return *(float*)&x;
    }

    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    float half_to_float(const unsigned short h) {
      return ((h&0x8000)<<16) | (((h&0x7c00)+0x1C000)<<13) | ((h&0x03FF)<<13);
    }

    // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    unsigned short float_to_half(const float f) {
      uint32_t x = *((uint32_t*)&f);
      return ((x>>16)&0x8000)|((((x&0x7f800000)-0x38000000)>>13)&0x7c00)|((x>>13)&0x03ff);
    }

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
          encode.set((uint)channels_, 2);
          return CommandEncoder::Size::From(input);
      });
      
      mem->get_contents(buffer_);
    }
    
    void CLut3DCopyFunction::foreach (std::function<void (uint, float, float, float)> block) {
      
      for (uint ib = 0; ib < lut_size_ ; ++ib) {
        for (uint ig = 0; ig < lut_size_ ; ++ig) {
          for (uint ir = 0; ir < lut_size_ ; ++ir) {
            auto index = static_cast<int>(ir + lut_size_ * ig + lut_size_ * lut_size_ * ib) * channels_;
            auto r = buffer_[index+0] ;
            auto g = buffer_[index+1] ;
            auto b = buffer_[index+2] ;
            block((uint)index, r, g, b);
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
