//
// Created by denn on 07.09.2021.
//

#include <string>
#include <iostream>
#include <fstream>
#include "dehancer/gpu/Lib.h"

#include "OpenColorIO/OpenColorIO.h"
#include "OpenColorIO/OpenColorTransforms.h"
//#include "OpenColorIO/transforms/Lut3DTransform.h"
//#include "OpenColorIO/OpBuilders.h"

namespace OCIO = OCIO_NAMESPACE;

int main(int argc, char **) {
  OCIO::ConstConfigRcPtr config    = OCIO::Config::Create();
  OCIO::FileTransformRcPtr cube_transform = OCIO::FileTransform::Create();
  cube_transform->setInterpolation(OCIO::INTERP_LINEAR);
  cube_transform->setDirection(OCIO::TRANSFORM_DIR_FORWARD);
  return 0;
}