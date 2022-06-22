//
// Created by denn on 25.04.2021.
//

#include "dehancer/gpu/clut/CLutTransformFunction.h"

namespace dehancer {
    CLutTransformFunction::CLutTransformFunction(const void *command_queue,
                                                 const std::string &kernel_name,
                                                 const Texture &source,
                                                 const Texture &target,
                                                 bool wait_until_completed,
                                                 const std::string &library_path)
            :
            Function(command_queue, kernel_name, wait_until_completed,library_path)
    {
  
      execute([&source,&target](CommandEncoder& encoder) {
          encoder.set(target,0);
          encoder.set(target,1);
          encoder.set(source,2);
          return CommandEncoder::Size::From(target);
      });
      
    }
  
}