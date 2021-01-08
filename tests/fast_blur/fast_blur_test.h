//
// Created by denn on 02.01.2021.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

//#define TEST_RADIUS 20
const float TEST_RADIUS[] = {90,90,90,0};
#define TEST_RADIUS_BOXED 1

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
  
  if (radius<=0) return 0;
  
  auto size = (int)ceil(radius/2+1) * 4 - 1;
  if (size%2==0) size+=1;
  if (size<3) size=3;
  
  std::cout << " Kernel size: " << size << std::endl;
  
  //weights.clear();
  //offsets.clear();
  
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


namespace dehancer {
    
    class UnaryKernel: public ChannelsInput {
    public:
        
        /**
         * Separable Row/Column function must be defined.
         * For example:
         *  box = 1/9 * [1 1 1 ...]' x [1 1 1 ...] is: 9x9 kernel weights matrix
         *                             1/9  ... 1/9
         *                             1/9  ... 1/9
         *                             ...      ...
         *                             1/9 ...  1/9
         */
        
        using KernelFunction = std::function<void (int channel_index, std::vector<float>& line)>;
        
        UnaryKernel(const void* command_queue,
                    const Texture& s,
                    const Texture& d,
                    const KernelFunction& row,
                    const KernelFunction& col,
                    bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                    const std::string& library_path = ""
        ):
                ChannelsInput (command_queue, s, wait_until_completed, library_path),
                row_func_(row),
                col_func_(col),
                w_(s->get_width()),
                h_(s->get_height()),
                channels_out_(ChannelsHolder::Make(command_queue,s->get_width(),s->get_height())),
                channels_finalizer_(command_queue, d, get_channels(), wait_until_completed)
        {
          
          for (int i = 0; i < 4; ++i) {
            std::vector<float> buf;
            
            row_func_(i,buf); row_sizes_[i] = buf.size();
            
            if (buf.empty())
              row_weights_[i] = nullptr;
            else
              row_weights_[i] =  dehancer::MemoryHolder::Make(get_command_queue(),
                                                              buf.data(),
                                                              buf.size()*sizeof(float ));
            
            buf.clear();
            
            col_func_(i,buf); col_sizes_[i] = buf.size();
            
            if (buf.empty())
              col_weights_[i] = nullptr;
            else
              col_weights_[i] =  dehancer::MemoryHolder::Make(get_command_queue(),
                                                              buf.data(),
                                                              buf.size()*sizeof(float ));
          }
        }
        
        void process() override {
          
          ChannelsInput::process();
          
          auto horizontal_kernel = Function(get_command_queue(),
                                            "convolve_horizontal_kernel");
          
          auto vertical_kernel = Function(get_command_queue(),
                                          "convolve_vertical_kernel",
                                          get_wait_completed());
          
          
          channels_finalizer_.process();
        }
    
    private:
        KernelFunction row_func_;
        KernelFunction col_func_;
        std::array<dehancer::Memory,4> row_weights_;
        std::array<int,4> row_sizes_;
        std::array<dehancer::Memory,4> col_weights_;
        std::array<int,4> col_sizes_;
        size_t w_;
        size_t h_;
        Channels channels_out_;
        ChannelsOutput channels_finalizer_;
    };
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
    
    int step_count[4]; //= make_fast_blur_convolve(TEST_RADIUS, host_weights, host_offsets);
    
    for (int i = 0; i < 4 ; ++i) {
      step_count[i] = make_fast_blur_convolve(TEST_RADIUS[i], host_weights, host_offsets);;
    }
    
    auto weights = dehancer::MemoryDesc {
            .length = host_weights.size() * sizeof(float)
    }.make(command_queue, host_weights.data());
    
    auto offsets = dehancer::MemoryDesc {
            .length = host_offsets.size() * sizeof(float )
    }.make(command_queue, host_offsets.data());
    
    auto step_count_array = dehancer::MemoryDesc {
            .length = 4 * sizeof(int)
    }.make(command_queue, &step_count[0]);
    
    std::chrono::time_point<std::chrono::system_clock> clock_begin
            = std::chrono::system_clock::now();
    
    kernel_convolve.execute([step_count_array,&tmp,&texture,&weights,&offsets](dehancer::CommandEncoder& command_encoder){
        command_encoder.set(texture,0);
        command_encoder.set(tmp,1);
        command_encoder.set(weights,2);
        command_encoder.set(offsets,3);
        
        command_encoder.set(step_count_array,4);
        int channels = 4;
        command_encoder.set(channels,5);
      
        dehancer::math::float2 direction = {1.0f,0.0f};
        command_encoder.set(direction,6);
        
        return dehancer::CommandEncoder::Size::From(tmp);
    });
    
    kernel_convolve.execute([step_count_array,&destination,&tmp,&weights,&offsets](dehancer::CommandEncoder& command_encoder){
        command_encoder.set(tmp,0);
        command_encoder.set(destination.get_texture(),1);
        command_encoder.set(weights,2);
        command_encoder.set(offsets,3);
        
        command_encoder.set(step_count_array,4);
        int channels = 4;
        command_encoder.set(channels,5);
        
        dehancer::math::float2 direction = {0.0f,1.0f};
        command_encoder.set(direction,6);
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
                                                 TEST_RADIUS_BOXED,TEST_RADIUS_BOXED,TEST_RADIUS_BOXED,TEST_RADIUS_BOXED
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