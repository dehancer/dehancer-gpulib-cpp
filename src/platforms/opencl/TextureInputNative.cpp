//
// Created by denn on 07.08.2022.
//

#include "dehancer/Log.h"
#include "platforms/TextureInput.h"

namespace dehancer::impl {
    #if defined(__APPLE__)
    Error TextureInput::load_from_native_image (const void *handle) {
      return Error(CommonError::NOT_SUPPORTED, "macos OpenCL API does not support native images loaded from native...");
    }
    #endif
}