//
// Created by denn on 26.04.2021.
//

#include "dehancer/gpu/profile/FilmProfile.h"
#include "dehancer/gpu/clut/CLutSquareInput.h"

namespace dehancer {
    
    FilmProfile::FilmProfile (const void *command_queue,
                              const StreamSpace &space,
                              StreamSpace::Direction direction,
                              bool wait_until_completed):
            Command(command_queue,wait_until_completed),
            space_(space),
            direction_(direction),
            cluts_()
    {}
    
    const CLut* FilmProfile::get (FilmProfile::Type type) const {
      if (cluts_[type])
        return cluts_[type].get();
      return nullptr;
    }
    
    Error FilmProfile::load (const MLutXmp &xmp) {
      
      for (int i = 0; i < xmp.get_cluts().size(); ++i) {
        if (i<cluts_.size()) {
          const auto& lut = xmp.get_cluts()[i];
          auto square_lut = CLutSquareInput(get_command_queue());
          auto error = square_lut.load_from_image(lut);
          if (error) return error;
          cluts_[i] = std::make_unique<CLutTransform>(
                  get_command_queue(),
                  square_lut,
                  dehancer::CLut::Type::lut_3d,
                  space_,
                  direction_
          );
        }
      }
      
      return Error(CommonError::OK);
    }
}