//
// Created by denn nevera on 29/11/2020.
//

#pragma once
#include "dehancer/gpu/Lib.h"
#include <chrono>

namespace dehancer {
    
    class ConvolveKernel: public ChannelsInput {
    public:
        
        ConvolveKernel(const void* command_queue,
                       const Texture& s,
                       const Texture& d,
                       std::array<float,4> radius,
                       bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                       const std::string& library_path = ""
        ):
                ChannelsInput (command_queue, s, wait_until_completed, library_path),
                radius_(radius),
                w_(s->get_width()),
                h_(s->get_height()),
                channels_out_(ChannelsHolder::Make(command_queue,s->get_width(),s->get_height())),
                channels_finalizer_(command_queue, d, get_channels(), wait_until_completed)
        {
          for (int i = 0; i < radius_.size(); ++i) {
            sizes_[i] = (int)ceil(radius_[i]/2.0f) * 6;
            if (sizes_[i]%2==0) sizes_[i]+=1;
            if (sizes_[i]<3) sizes_[i]=3;
            std::vector<float> ww;
            dehancer::math::make_gaussian_kernel(ww, sizes_[i], radius_[i]/2.0f);
            
            if (ww.empty())
              weights_[i] = nullptr;
            else
              weights_[i] =  dehancer::MemoryHolder::Make(get_command_queue(),
                                                          ww.data(),
                                                          ww.size()*sizeof(float ));
          }
        }
        
        void process() override {
          
          ChannelsInput::process();
          
          auto horizontal_kernel = Function(get_command_queue(),
                                            "convolve_horizontal_kernel",
                                            get_wait_completed());
          
          auto vertical_kernel = Function(get_command_queue(),
                                          "convolve_vertical_kernel",
                                          get_wait_completed());
          
          for (int i = 0; i < get_channels()->size(); ++i) {
            
            if (radius_[i]>0) {
              
              horizontal_kernel.execute([this, i](CommandEncoder &command) {
                  auto in = get_channels()->at(i);
                  auto out = channels_out_->at(i);
                  
                  command.set(in, 0);
                  command.set(out, 1);
                  
                  int w = w_, h = h_;
                  command.set(w, 2);
                  command.set(h, 3);
                  
                  command.set(weights_.at(i), 4);
                  command.set(sizes_[i], 5);
                  
                  return (CommandEncoder::Size) {w_ + sizes_[i] / 2, h_, 1};
              });
              
              vertical_kernel.execute([this, i](CommandEncoder &command) {
                  auto in = channels_out_->at(i);
                  auto out = get_channels()->at(i);
                  
                  command.set(in, 0);
                  command.set(out, 1);
                  
                  int w = w_, h = h_;
                  command.set(w, 2);
                  command.set(h, 3);
                  
                  command.set(weights_.at(0), 4);
                  command.set(sizes_[0], 5);
                  
                  return (CommandEncoder::Size) {w_, h_ + sizes_[i] / 2, 1};
              });
            }
          }
          
          channels_finalizer_.process();
        }
    
    private:
        std::array<float,4> radius_;
        std::array<dehancer::Memory,4> weights_;
        std::array<int,4> sizes_;
        size_t w_;
        size_t h_;
        Channels channels_out_;
        ChannelsOutput channels_finalizer_;
    };
}


int run_on_device(int num, const void* device, std::string patform) {
  
  dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
  std::string ext = dehancer::TextureIO::extention_for(type);
  float compression = 0.3f;
  
  size_t width = 800*2, height = 400*2 ;
  
  auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));
  
  auto grid_kernel = dehancer::Function(command_queue,"kernel_grid");
  auto grid_text = grid_kernel.make_texture(width, height);
  
  /**
   * Debug info
   */
  
  std::cout << "[grid kernel " << grid_kernel.get_name() << " args: " << std::endl;
  for (auto &a: grid_kernel.get_arg_list()) {
    std::cout << std::setw(20) << a.name << "[" << a.index << "]: " << a.type_name << std::endl;
  }
  
  /***
   * Test performance
   */
  grid_kernel.execute([&grid_text](dehancer::CommandEncoder& command_encoder){
      int levels = 6;
      
      command_encoder.set(levels, 0);
      command_encoder.set(grid_text, 1);
      
      return dehancer::CommandEncoder::Size::From(grid_text);
  });
  
  std::string out_file_cv = "grid-"+patform+"-"; out_file_cv.append(std::to_string(num)); out_file_cv.append(ext);
  
  {
    std::ofstream ao_bench_os(out_file_cv, std::ostream::binary | std::ostream::trunc);
    ao_bench_os << dehancer::TextureOutput(command_queue, grid_text, {
            .type = type,
            .compression = compression
    });
  }
  
  
  auto output_text = dehancer::TextureOutput(command_queue, width, height, nullptr, {
          .type = type,
          .compression = compression
  });
  
  auto blur_line_kernel = dehancer::ConvolveKernel(command_queue,
                                                   grid_text,
                                                   output_text.get_texture(),
                                                   {15,15,15,1},
                                                   true
  );
  
  std::cout << "[convolve_line_kernel kernel " << grid_kernel.get_name() << " args: " << std::endl;
  for (auto &a: blur_line_kernel.get_arg_list()) {
    std::cout << std::setw(20) << a.name << "[" << a.index << "]: " << a.type_name << std::endl;
  }
  
  std::chrono::time_point<std::chrono::system_clock> clock_begin
          = std::chrono::system_clock::now();
  
  blur_line_kernel.process();
  
  std::chrono::time_point<std::chrono::system_clock> clock_end
          = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = clock_end-clock_begin;
  
  // Report results and save image
  auto device_type = dehancer::device::get_type(device);
  
  std::string device_type_str;
  
  switch (device_type) {
    case dehancer::device::Type::cpu :
      device_type_str = "CPU"; break;
    case dehancer::device::Type::gpu :
      device_type_str = "GPU"; break;
    default:
      device_type_str = "Unknown"; break;
  }
  
  std::cout << "[convolve-processing "
            <<patform<<"/"<<device_type_str
            <<" ("
            <<dehancer::device::get_name(device)
            <<")]:\t" << seconds.count() << "s "
            << ", for a " << width << "x" << height << " pixels" << std::endl;
  
  
  std::string out_file_result = "blur-line-"+patform+"-result-"; out_file_result.append(std::to_string(num)); out_file_result.append(ext);
  {
    std::ofstream result_os(out_file_result, std::ostream::binary | std::ostream::trunc);
    result_os << output_text;
  }
  
  std::chrono::time_point<std::chrono::system_clock> clock_io_end
          = std::chrono::system_clock::now();
  seconds = clock_io_end-clock_end;
  
  std::cout << "[convolve-output     "
            <<patform<<"/"<<device_type_str
            <<" ("
            <<dehancer::device::get_name(device)
            <<")]:\t" << seconds.count() << "s "
            << ", for a " << width << "x" << height << " pixels" << std::endl;
  
  return 0;
}

void run(std::string platform) {
  try {
    
    std::vector<float> g_kernel;
    
    float radius = 0.5;
    float sigma  = radius/2.0;
    int size = (int)ceilf(6 * sigma);
    if (size<=2) size = 3;
    if (size % 2 == 0) size++;
    
    dehancer::math::make_gaussian_kernel(g_kernel, size, sigma);
    
    for (int i = 0; i < g_kernel.size(); ++i) {
      std::cout << " kernel weight["<<i<<"] = " << g_kernel[i] << std::endl;
    }
    
    auto devices = dehancer::DeviceCache::Instance().get_device_list(dehancer::device::Type::gpu);
    assert(!devices.empty());
    
    int dev_num = 0;
    std::cout << "Platform: " << platform << std::endl;
    for (auto d: devices) {
      std::cout << " #" << dev_num++ << std::endl;
      std::cout << "    Device '"
                << dehancer::device::get_name(d)
                << " ["<<dehancer::device::get_id(d)<<"]'"<< std::endl;
    }
    
    std::cout << "Bench: " << std::endl;
    dev_num = 0;
    
    for (auto d: devices) {
#if __APPLE__
      if (dehancer::device::get_type(d) == dehancer::device::Type::cpu) continue;
#endif
      if (run_on_device(dev_num++, d, platform) != 0) return;
    }
    
  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}