//
// Created by denn nevera on 29/11/2020.
//

#pragma once
#include "dehancer/gpu/Lib.h"
#include <chrono>
#include <vector>
#include <any>

#include "tests/test_config.h"

const float TEST_RADIUS[]     = {20,0,0,0};
const int TEST_BOX_RADIUS[]   = {4,4,4,0};
const float TEST_RESOLURION[] = {3.8,3.8,3.8,0};

static dehancer::ChannelDesc::Transform options_one = {
        .slope   = {8.0f,  8.0f, 8,0},
        .offset  = {16.0f, 16.0f, 16,0},
        .enabled = {true,true,false,false},
        .direction = dehancer::ChannelDesc::TransformDirection::forward
};

namespace test {
    class Convolver: public dehancer::UnaryKernel {
    public:
        using dehancer::UnaryKernel::UnaryKernel;
        
        Convolver(const void* command_queue):
                dehancer::UnaryKernel(command_queue, {}, {}, true) {};
        
        void set_options(const Options &options) override {
          dehancer::UnaryKernel::set_options(options);
        }
    };
}

void downscale_kernel (int length_, std::vector<float>& kernel) {
  float step = length_;
  int length = ceil(step);
  int mid = length*3/2;
  kernel.resize(3*length);
  for (int i=0; i<=length/2; i++) {
    double x = i/(double)step;
    float v = (float)((0.75-x*x)/length);
    kernel[mid-i] = v;
    kernel[mid+i] = v;
  }
  for (int i=length/2+1; i<(length*3+1)/2; i++) {
    double x = i/(double)step;
    float v = (float)((0.125 + 0.5*(x-1)*(x-2))/length);
    kernel[mid-i] = v;
    kernel[mid+i] = v;
  }
}

void magic_resampler(float length_, std::vector<float>& kernel) {
  int size = ceil(3/2*length_);
  int half_size = ceil((float)size/2.0f+1)+1;
  
  if(half_size%2==0) half_size+=1;
  float sum = 0;
  for (int i = -half_size; i <= half_size; ++i) {
    float x = (float )i;
    if      ( x <= -3.0f/2.0f*length_ ) x = 0;
    else if ( x >  -3.0f/2.0f*length_ && x <= -1.0f/2.0f*length_ ) x = 1.0f/2.0f*pow(x+3.0/2.0*length_,2.0f);
    else if ( x >  -1.0/2.0*length_   && x <   1.0f/2.0f*length_ ) x = 3.0f/4.0f*length_-x*x;
    else if ( x >=  1.0/2.0*length_   && x <   3.0f/2.0f*length_ ) x = 1.0f/2.0f*pow(x-3.0/2.0*length_,2.0f);
    else if ( x >= -3.0f/2.0f*length_ ) x = 0;
    kernel.push_back(x);
    sum += x;
  }
  
  for (auto& v: kernel) {
    v/=sum;
  }
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
  
  auto kernel_blur = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
      data.clear();
      
      auto radius = TEST_RADIUS[index];
      
      if (radius==0) return ;
      
      float sigma = radius/2.0f;
      int kRadius = (int)ceil(sigma*sqrt(-2.0f*log(0.0001)))+1;
      
      auto size = kRadius;// (int)ceil(radius/2+1) * 4 - 1;
      if (size%2==0) size+=1;
      if (size<3) size=3;
      dehancer::math::make_gaussian_kernel(data, size, radius/2.0f);
  };
  
  
  auto kernel_magic_resolution = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
      data.clear();
      if (index==3) return ;
      float r = TEST_RESOLURION[index];
      if (r==0) return;
      magic_resampler(r,data);
  };
  
  auto kernel_resample = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
      data.clear();
      if (index==3) return ;
      size_t size = 2;
      downscale_kernel(size,data);
      data.erase(data.begin());
      int i = 0;
      for (auto v: data) {
        std::cout << "d["<<i++<<"] = " << v << std::endl;
      }
  };
  
  auto kernel_box_blur = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
      data.clear();
      if (index==3) return ;
      int radius = TEST_BOX_RADIUS[index];
      if (radius <= 1 ) return;
      for (int i = 0; i < radius; ++i) {
        data.push_back(1.0f/(float)radius);
      }
  };
  
  struct kernel_funcx {
      dehancer::UnaryKernel::KernelFunction row, col;
      std::string name;
  };
  
  std::vector<kernel_funcx> kernels = {
          {
                  .row = kernel_blur,
                  .col = kernel_blur,
                  .name = "blur"
          },
          {
                  .row = kernel_magic_resolution,
                  .col = kernel_magic_resolution,
                  .name = "resolution"
          },
          {
                  .row = kernel_box_blur,
                  .col = kernel_box_blur,
                  .name = "box-blur"
          }
        
          ,
          {
                  .row = kernel_resample,
                  .col = kernel_resample,
                  .name = "resampler"
          }
  };
  
  auto lena_text = dehancer::TextureInput(command_queue);
  std::string lena_file = IMAGES_DIR; lena_file +="/"; lena_file+= IMAGE_FILES[5];
  
  std::ifstream ifs(lena_file, std::ios::binary);
  ifs >> lena_text;
  
  std::vector<dehancer::Texture> inputs = {grid_text,lena_text.get_texture()};
  
  auto line_kernel = test::Convolver(command_queue);
  
  auto grad_kernel = dehancer::Function(command_queue,"kernel_gradient");
  auto grad_text = grad_kernel.make_texture(800, 400);
  
  grad_kernel.execute([&grad_text](dehancer::CommandEncoder& command_encoder){
      command_encoder.set(grad_text, 0);
      command_encoder.set(false, 1);
      return dehancer::CommandEncoder::Size::From(grad_text);
  });
  
  {
    std::ofstream gfs("gradient."+ext, std::ios::binary);
    gfs << dehancer::TextureOutput(command_queue, grad_text, {
            .type = type,
            .compression = compression
    });
  }
  
  options_one.mask = grad_text;
  
  line_kernel.set_transform(options_one);
  
  for (auto kf: kernels) {
    int text_num = 0;
    
    for (auto text: inputs) {
      
      auto output_text = dehancer::TextureOutput(command_queue, text->get_width(), text->get_height(), nullptr, {
              .type = type,
              .compression = compression
      });
      
      std::cout << "[convolve_line_kernel kernel " << grid_kernel.get_name() << " args: " << std::endl;
      for (auto &a: line_kernel.get_arg_list()) {
        std::cout << std::setw(20) << a.name << "[" << a.index << "]: " << a.type_name << std::endl;
      }
      
      std::chrono::time_point<std::chrono::system_clock> clock_begin
              = std::chrono::system_clock::now();
      
      dehancer::UnaryKernel::Options options =  {
              .row = kf.row,
              .col = kf.col,
              .user_data = kf.name,
              .edge_mode = DHCR_EdgeMode ::DHCR_ADDRESS_CLAMP,
      };
      
      
      line_kernel.set_options(options);
      
      //line_kernel.set_source(text);
      //line_kernel.set_destination(output_text.get_texture());
      
      line_kernel.process(text, output_text.get_texture());
      
      std::chrono::time_point<std::chrono::system_clock> clock_end
              = std::chrono::system_clock::now();
      std::chrono::duration<double> seconds = clock_end - clock_begin;
      
      // Report results and save image
      auto device_type = dehancer::device::get_type(device);
      
      std::string device_type_str;
      
      switch (device_type) {
        case dehancer::device::Type::cpu :
          device_type_str = "CPU";
          break;
        case dehancer::device::Type::gpu :
          device_type_str = "GPU";
          break;
        default:
          device_type_str = "Unknown";
          break;
      }
      
      std::cout << "[convolve-" << kf.name << "-processing "
                << patform << "/" << device_type_str
                << " ("
                << dehancer::device::get_name(device)
                << ")]:\t" << seconds.count() << "s "
                << ", for a " << width << "x" << height << " pixels" << std::endl;
      
      
      std::string out_file_result = kf.name + "-line-" + patform + "-result-";
      out_file_result.append(std::to_string(text_num));
      out_file_result.append("-");
      out_file_result.append(std::to_string(num));
      out_file_result.append(ext);
      {
        std::ofstream result_os(out_file_result, std::ostream::binary | std::ostream::trunc);
        result_os << output_text;
      }
      
      std::chrono::time_point<std::chrono::system_clock> clock_io_end
              = std::chrono::system_clock::now();
      seconds = clock_io_end - clock_end;
      
      std::cout << "[convolve-" << kf.name << "-output-"<<text_num<<"-     "
                << patform << "/" << device_type_str
                << " ("
                << dehancer::device::get_name(device)
                << ")]:\t" << seconds.count() << "s "
                << ", for a " << width << "x" << height << " pixels" << std::endl;
      text_num++;
    }
  }
  
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