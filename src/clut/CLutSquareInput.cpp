//
// Created by denn on 26.04.2021.
//

#include "dehancer/gpu/clut/CLutSquareInput.h"

namespace dehancer {
    
    CLutSquareInput::CLutSquareInput (const void *command_queue//,
                                      //const StreamSpace &space,
                                      //StreamSpaceDirection direction
                                      ) :
            TextureInput(command_queue),// space,direction),
            lut_size_(0)
    {
    }
    
    Error
    CLutSquareInput::load_from_data (const std::vector<float> &buffer, size_t width, size_t height, size_t depth) {
      auto e =  TextureInput::load_from_data(buffer, width, height, depth);
      if (!e) {
        update_lut_size();
      }
      return e;
    }
    
    Error CLutSquareInput::load_from_data (const std::vector<float> &buffer, size_t width, size_t height) {
      auto e =  TextureInput::load_from_data(buffer, width, height);
      if (!e) {
        update_lut_size();
      }
      return e;
    }
    
    Error CLutSquareInput::load_from_image (const std::vector<uint8_t> &buffer) {
      auto e =  TextureInput::load_from_image(buffer);
      if (!e) {
        update_lut_size();
      }
      return e;
    }
    
    Error CLutSquareInput::load_from_image (const uint8_t *buffer, size_t length) {
      auto e = TextureInput::load_from_image(buffer, length);
      if (!e) {
        update_lut_size();
      }
      return e;
    }
    
    Error CLutSquareInput::load_from_data (float *buffer, size_t width, size_t height, size_t depth) {
      auto e = TextureInput::load_from_data(buffer, width, height, depth);
      if (!e) {
        update_lut_size();
      }
      return e;
    }
    
    void CLutSquareInput::update_lut_size () {
      if (get_texture()) {
        auto level = (uint) std::round(powf((float) get_texture()->get_width(), 1.0f / 3.0f));
        lut_size_ = (uint) std::round(level * level);
      }
      else
        lut_size_ = 0;
    }
    
    const Texture &CLutSquareInput::get_texture () {
      return TextureInput::get_texture();
    }
    
    const Texture &CLutSquareInput::get_texture () const {
      return TextureInput::get_texture();
    }
    
}