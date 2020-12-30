//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"

#include <chrono>

dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
[[maybe_unused]] std::string ext = dehancer::TextureIO::extention_for(type);
float       compression = 0.3f;

void load_texture(const std::string& path, void *command_queue, int num) {

  std::cout << "Load file: " << path << std::endl;

  auto input_text = dehancer::TextureInput(command_queue);

  std::ifstream ifs(path, std::ios::binary);
  ifs >> input_text;

  std::string out_file_cv = "cuda-texture-io-"; out_file_cv.append(std::to_string(num)); out_file_cv.append(ext);

  {
    std::ofstream os(out_file_cv, std::ostream::binary | std::ostream::trunc);
    if (os.is_open()) {
      os << dehancer::TextureOutput(command_queue, input_text.get_texture(), {
              .type = type,
              .compression = compression
      }) << std::flush;

      std::cout << "Save to: " << out_file_cv << std::endl;

    } else {
      std::cerr << "File: " << out_file_cv << " could not been opened..." << std::endl;
    }
  }
}

TEST(TEST, CUDA_TEXTURE) {

  std::cout << std::endl;
  std::cerr << std::endl;

  auto* command_queue = dehancer::DeviceCache::Instance().get_default_command_queue();

  int i = 0;
  for (auto& file: IMAGE_FILES) {
    std::string path = CUDA_IMAGE_DIR; path.append("/"); path.append(file);
    load_texture(path,command_queue,i++);
  }


  dehancer::DeviceCache::Instance().return_command_queue(command_queue);

}

namespace dehancer::device {

    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      return CUDA_KERNELS_LIBRARY;// + std::string("++");
    }
}