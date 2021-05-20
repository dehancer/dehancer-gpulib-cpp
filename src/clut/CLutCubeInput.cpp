//
// Created by denn nevera on 08/06/2020.
//

#include "dehancer/gpu/clut/CLutCubeInput.h"
#include "dehancer/gpu/clut/utils/CubeParser.h"

namespace dehancer {
    
    CLutCubeInput::CLutCubeInput(const void *command_queue):
            CLut(),
            TextureInput(command_queue),
            lut_size_(0)
    {
    }
    
    CLutCubeInput::~CLutCubeInput() = default;
    
    std::istream &operator>>(std::istream &is, CLutCubeInput &dt) {
      CubeParser parser;
      
      is >> parser;
      
      dt.load_from_data(parser.get_lut(), parser.get_lut_size());
      
      return is;
    }
    
    const dehancer::Texture &dehancer::CLutCubeInput::get_texture () {
      return TextureInput::get_texture();
    }
    
    const Texture &CLutCubeInput::get_texture () const {
      return TextureInput::get_texture();
    }
    
    Error CLutCubeInput::load_from_data (float *buffer, size_t lut_size) {
      lut_size_ = lut_size;
      return TextureInput::load_from_data(buffer, lut_size, lut_size, lut_size);
    }
    
    Error CLutCubeInput::load_from_data (const std::vector<float> &buffer, size_t lut_size) {
      lut_size_ = lut_size;
      return TextureInput::load_from_data(buffer, lut_size, lut_size, lut_size);
    }
  
}