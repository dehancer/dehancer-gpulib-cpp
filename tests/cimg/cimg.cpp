//
// Created by denn nevera on 09/11/2020.
//


#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include "../src/external/CImg.h"

TEST(USER, CIMG) {

  std::cout << std::endl;
  std::cerr << std::endl;

  using namespace cimg_library;

  const unsigned char white[] = { 255,255,255 };

  const CImg<unsigned char> img
          = CImg<unsigned char>(800,600,1,3)
                  .fill(32)
                  .noise(128)
                  .blur(2)
                  .draw_text(400,300,"Hello World", white,0,80);

  img.save_png("cimg_test.png");
  img.save_tiff("cimg_test.tiff");
  img.save_jpeg("cimg_test.jpeg", 50);

  CImg<float> img_loaded;//(400*3*300); //"cimg_test.tiff"

  img_loaded.load_tiff("cimg_test.tiff");
  //img_loaded.assign("cimg_test.tiff");

  img_loaded.save_png("cimg_test_from_float.png");
  //img_loaded.data();
}