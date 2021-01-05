//
// Created by denn on 02.01.2021.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

#define TEST_RADIUS 90

inline std::string stringFormatA( const char * fmt, ... )
{
  int nSize = 0;
  char buff[4096];
  va_list args;
  va_start(args, fmt);
  nSize = vsnprintf( buff, sizeof(buff) - 1, fmt, args); // C4996
  return std::string( buff );
}

int make_fast_blur_convolve(float radius, std::vector<float>& weights, std::vector<float>& offsets) {
  
  auto size = (int)ceil(radius/2+1) * 4 - 1;
  if (size%2==0) size+=1;
  if (size<3) size=3;
  
  std::cout << " Kernel size: " << size << std::endl;
  
  weights.clear();
  offsets.clear();
  
  std::vector<float> inputKernel;
  dehancer::math::make_gaussian_kernel(inputKernel, size, radius/2.0f);
  
  std::vector<float> oneSideInputs;
  for( int i = (size/2); i >= 0; i-- )
  {
    if( i == (size/2) )
      oneSideInputs.push_back( (float)inputKernel[i] * 0.5f );
    else
      oneSideInputs.push_back( (float)inputKernel[i] );
  }
  
  assert( (oneSideInputs.size() % 2) == 0 );
  int numSamples = oneSideInputs.size()/2;
  
  
  for( int i = 0; i < numSamples; i++ )
  {
    float sum = oneSideInputs[i*2+0] + oneSideInputs[i*2+1];
    weights.push_back(sum);
  }
  
  for( int i = 0; i < numSamples; i++ )
  {
    offsets.push_back( i*2.0f + oneSideInputs[i*2+1] / weights[i] );
  }
  
  return numSamples;
}

auto fast_blur_test =  [] (int dev_num,
                           const void* command_queue,
                           const dehancer::Texture& texture,
                           const std::string& platform) {
    
    std::cout << "Test fast blur on platform: " << platform << std::endl;
    
    auto destination = dehancer::TextureOutput(command_queue, texture, {
            .type =  dehancer::TextureOutput::Options::Type::png,
            .compression = 0.3f
    });
    
    auto tmp = dehancer::TextureDesc{
            .width = destination.get_texture()->get_width(),
            .height = destination.get_texture()->get_height()
    }.make(command_queue);
    
    auto kernel_convolve = dehancer::Function(command_queue, "kernel_fast_convolve", true);
    
    std::vector<float> host_weights, host_offsets;
    
    auto stepCount = make_fast_blur_convolve(TEST_RADIUS, host_weights, host_offsets);
    
    auto weights = dehancer::MemoryDesc {
            .length = host_weights.size() * sizeof(float)
    }.make(command_queue, host_weights.data());
    
    auto offsets = dehancer::MemoryDesc {
            .length = host_offsets.size() * sizeof(float )
    }.make(command_queue, host_offsets.data());
    
    std::chrono::time_point<std::chrono::system_clock> clock_begin
            = std::chrono::system_clock::now();
    
    kernel_convolve.execute([stepCount,&tmp,&texture,&weights,&offsets](dehancer::CommandEncoder& command_encoder){
        command_encoder.set(texture,0);
        command_encoder.set(tmp,1);
        command_encoder.set(weights,2);
        command_encoder.set(offsets,3);
        command_encoder.set(stepCount,4);
        dehancer::math::float2 direction = {1.0f,0.0f};
        command_encoder.set(direction,5);
        return dehancer::CommandEncoder::Size::From(tmp);
    });
    
    kernel_convolve.execute([stepCount,&destination,&tmp,&weights,&offsets](dehancer::CommandEncoder& command_encoder){
        command_encoder.set(tmp,0);
        command_encoder.set(destination.get_texture(),1);
        command_encoder.set(weights,2);
        command_encoder.set(offsets,3);
        command_encoder.set(stepCount,4);
        dehancer::math::float2 direction = {0.0f,1.0f};
        command_encoder.set(direction,5);
        return dehancer::CommandEncoder::Size::From(destination.get_texture());
    });
    
    std::chrono::time_point<std::chrono::system_clock> clock_end
            = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = clock_end-clock_begin;
    
    {
      
      std::string out_file_cv = "fast-blur-io-";
      out_file_cv.append(platform);
      out_file_cv.append("-["); out_file_cv.append(std::to_string(dev_num)); out_file_cv.append("]");
      out_file_cv.append(test::ext);
      
      std::ofstream os(out_file_cv, std::ostream::binary | std::ostream::trunc);
      if (os.is_open()) {
        os << destination << std::flush;
        
        std::cout << "Save to: " << out_file_cv << std::endl;
        
      } else {
        std::cerr << "File: " << out_file_cv << " could not been opened..." << std::endl;
      }
    }
    
    std::cout << "[convolve-processing "
              <<platform<<"/"<<"fast-blur"
              <<" ("
              <<"-"
              <<")]:\t" << seconds.count() << "s "
              << ", for a " << texture->get_width() << "x" << texture->get_height() << " pixels" << std::endl;
    
    return 0;
};


auto gaussian_boxed_blur_test =  [] (int dev_num,
                                     const void* command_queue,
                                     const dehancer::Texture& texture,
                                     const std::string& platform) {
    
    std::cout << "Test fast blur on platform: " << platform << std::endl;
    
    auto destination = dehancer::TextureOutput(command_queue, texture, {
            .type =  dehancer::TextureOutput::Options::Type::png,
            .compression = 0.3f
    });
    
    auto kernel = dehancer::GaussianBlur(command_queue, texture, destination.get_texture(),
                                         {
                                                 TEST_RADIUS,TEST_RADIUS,TEST_RADIUS,TEST_RADIUS
                                         }, true);
    
    std::chrono::time_point<std::chrono::system_clock> clock_begin
            = std::chrono::system_clock::now();
    
    kernel.process();
    
    std::chrono::time_point<std::chrono::system_clock> clock_end
            = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = clock_end-clock_begin;
    
    {
      
      std::string out_file_cv = "gaussian-boxed-blur-io-";
      out_file_cv.append(platform);
      out_file_cv.append("-["); out_file_cv.append(std::to_string(dev_num)); out_file_cv.append("]");
      out_file_cv.append(test::ext);
      
      std::ofstream os(out_file_cv, std::ostream::binary | std::ostream::trunc);
      if (os.is_open()) {
        os << destination << std::flush;
        
        std::cout << "Save to: " << out_file_cv << std::endl;
        
      } else {
        std::cerr << "File: " << out_file_cv << " could not been opened..." << std::endl;
      }
    }
    
    std::cout << "[convolve-processing "
              <<platform<<"/"<<"gaussian-boxed-blur"
              <<" ("
              <<"-"
              <<")]:\t" << seconds.count() << "s "
              << ", for a " << texture->get_width() << "x" << texture->get_height() << " pixels" << std::endl;
  
    return 0;
};