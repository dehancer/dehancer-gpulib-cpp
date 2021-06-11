//
// Created by denn on 26.04.2021.
//

#include "dehancer/gpu/profile/CameraProfile.h"
#include "dehancer/gpu/clut/CLutSquareInput.h"

namespace dehancer {
    
    CameraProfile::CameraProfile (const void *command_queue,
                                  const StreamSpace &space,
                                  StreamSpaceDirection direction,
                                  bool wait_until_completed):
            Command(command_queue,wait_until_completed),
            space_(space),
            direction_(direction),
            clut_(nullptr)
    {}
    
    const std::shared_ptr<CLut>& CameraProfile::get () const {
      return clut_;
    }
    
    Error CameraProfile::load (const CameraLutXmp &xmp) {
      
      const auto& lut = xmp.get_clut();
      auto square_lut = CLutSquareInput(get_command_queue());
      auto error = square_lut.load_from_image(lut);
      if (error) return error;
      
      clut_ = std::make_unique<CLutTransform>(
              get_command_queue(),
              square_lut,
              dehancer::CLut::Type::lut_3d,
              0,
              space_,
              direction_
      );
      
      return Error(CommonError::OK);
    }
}