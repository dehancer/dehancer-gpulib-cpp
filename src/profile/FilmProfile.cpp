//
// Created by denn on 26.04.2021.
//

#include "dehancer/gpu/profile/FilmProfile.h"
#include "dehancer/gpu/clut/CLutSquareInput.h"
#include "dehancer/Log.h"

namespace dehancer {
    
    FilmProfile::FilmProfile (const void *command_queue,
                              CLut::Type type,
                              const StreamSpace& space,
                              StreamSpaceDirection direction,
                              bool wait_until_completed):
            Command(command_queue,wait_until_completed),
            space_(space),
            direction_(direction),
            cluts_(),
            type_(type)
    {}
    
    const std::shared_ptr<CLut>& FilmProfile::get (FilmProfile::Type type) const {
      return cluts_[type];
    }
    
    Error FilmProfile::load (const MLutXmp &xmp) {
      
      for (size_t i = 0; i < xmp.get_cluts().size(); ++i) {
        if (i<cluts_.size()) {
          const auto& lut = xmp.get_cluts()[i];
          auto square_lut = CLutSquareInput(get_command_queue());
          
          auto error = square_lut.load_from_image(lut);
          if (error) return error;
          cluts_[i] = std::make_unique<CLutTransform>(
                  get_command_queue(),
                  square_lut,
                  type_,
                  65,
                  space_,
                  direction_
          );
          dehancer::log::print("FilmProfile::load lut size = %zu, length = %zuMb.", cluts_[i]->get_lut_size(), cluts_[i]->get_texture()->get_length()/1024/1024);
        }
      }
      
      return Error(CommonError::OK);
    }
    
    FilmProfile::FilmProfile (const void *command_queue, const StreamSpace &space,
                              StreamSpaceDirection direction, bool wait_until_completed):
            FilmProfile (command_queue,CLut::Type::lut_3d,space,direction,wait_until_completed)
    {
    }
}