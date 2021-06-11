//
// Created by denn nevera on 2019-07-21.
//

#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include "dehancer/MLutXmp.h"
#include "gtest/gtest.h"

#include "./dotenv.h"

std::vector<std::uint8_t > split_by_pattern(const std::string& content, const std::string& pattern){

  auto p = std::regex(pattern);
  std::vector<uint8_t> floats;

  std::transform(
          std::sregex_token_iterator(content.begin(), content.end(), p, -1),
          std::sregex_token_iterator(), back_inserter(floats),
          [](const std::string& name){
              return std::stof(name);
          });

  return floats;
}


Blowfish::KeyType get_key() {
  auto clut_key = dotenv::get_dotenv("CMLUT_KEY");

  EXPECT_TRUE(clut_key);

  if (!clut_key) {
    std::cerr << "CMLUT_KEY: " << clut_key.error().message() << std::endl;
    return {};
  }

  return split_by_pattern(clut_key.value(), "\\s*,\\s*");
}

std::string get_clut_file_path() {
  auto path = dotenv::get_dotenv("CLUT_FILE");
  EXPECT_TRUE(path);
  if (!path) {
    std::cerr << "CMLUT_KEY: " << path.error().message() << std::endl;
    return "";
  }
  return path.value();
}

std::string get_dehancerd_url() {
  auto path = dotenv::get_dotenv("DEHANCERD_URL");
  EXPECT_TRUE(path);
  if (!path) {
    std::cerr << "DEHANCERD_URL: " << path.error().message() << std::endl;
    return "";
  }
  return path.value();
}