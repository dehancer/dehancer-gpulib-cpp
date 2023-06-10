//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"
#include "tests/test_config.h"

#include <chrono>

TEST(TEST, CUDA_EMBEDED_FATBIN) {

  std::cout << std::endl;
  std::cerr << std::endl;

  auto* command_queue = dehancer::DeviceCache::Instance().get_default_command_queue();
  
  auto grid_kernel = dehancer::Function(command_queue,"kernel_grid");
  auto source = grid_kernel.make_texture(1024, 1024);
  
  /***
   * Test performance
   */
  grid_kernel.execute([&source](dehancer::CommandEncoder& command_encoder){
      int levels = 6;
      
      command_encoder.set(levels, 0);
      command_encoder.set(source, 1);
      
      return dehancer::CommandEncoder::Size::From(source);
  });
  
  auto kernel = dehancer::BoxBlur(command_queue);
  
  auto output_text = dehancer::TextureOutput(command_queue, 1024, 1024, {
          .type = test::type,
          .compression = test::compression
  });
  
  kernel.set_source(source);
  kernel.set_destination(output_text.get_texture());
  kernel.set_radius(20);
  
  kernel.process();
  
  std::ostringstream osss; osss << "cuda-embedded" << "." << dehancer::TextureIO::extension_for(test::type);
  
  std::ofstream os(osss.str(), std::ostream::binary | std::ostream::trunc);
  
  if (os.is_open()) {
    os << output_text << std::flush;
    std::cout << "Save to: " << osss.str() << std::endl;
    
  } else {
    std::cerr << "File: " << osss.str() << " could not been opened..." << std::endl;
  }
  
  dehancer::DeviceCache::Instance().return_command_queue(command_queue);
  
  
}
