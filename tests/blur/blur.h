//
// Created by denn nevera on 29/11/2020.
//

#pragma once
#include "dehancer/gpu/Lib.h"
#include <chrono>
#include <vector>
#include <set>
#include <cmath>

namespace dehancer::math {

    using KernelLine = std::pair<std::vector<float>, std::vector<float>>;

    void make_gaussian_kernel(KernelLine &kernel,
                              float sigma,
                              float accuracy,
                              int maxRadius = 50) {
      int kRadius = (int) std::ceil(sigma * std::sqrt(-2.0 * std::log(accuracy))) + 1.0;
      if (maxRadius < 16) maxRadius = 16;         // too small maxRadius would result in inaccurate sum.
      if (kRadius > maxRadius) kRadius = maxRadius;

      kernel.first.resize(kRadius);
      kernel.second.resize(kRadius);

      for (int i=0; i<kRadius; i++)   // Gaussian function
        kernel.first[i] = (float)(std::exp(-0.5*i*i/sigma/sigma));

      if (kRadius < maxRadius && kRadius > 3) {   // edge correction
        float sqrtSlope = FLT_MAX;
        int r = kRadius;
        while (r > kRadius/2) {
          r--;
          float a = std::sqrt(kernel.first[r])/(kRadius-r);
          if (a < sqrtSlope)
            sqrtSlope = a;
          else
            break;
        }
        for (int r1 = r+2; r1 < kRadius; r1++)
          kernel.first[r1] = (float)((kRadius-r1)*(kRadius-r1)*sqrtSlope*sqrtSlope);
      }

      float sum = 0; // sum over all kernel elements for normalization
      if (kRadius < maxRadius) {
        sum = kernel.first[0];
        for (int i=1; i<kRadius; i++)
          sum += 2*kernel.first[i];
      } else
        sum = sigma * std::sqrt(2*M_PI);

      float rsum = 0.5 + 0.5*kernel.first[0]/sum;
      for (int i=0; i<kRadius; i++) {
        float v = (kernel.first[i]/sum);
        kernel.first[i] = (float)v;
        rsum -= v;
        kernel.second[i] = (float)rsum;
      }
      std::cerr << " ... kRadius = " << kRadius << std::endl;
    }

    ///
    /// http://blog.ivank.net/fastest-gaussian-blur.html
    ///
    void make_gauss_boxes(std::vector<float>& boxes, float sigma, size_t box_number) {
      float n = static_cast<float>(box_number);
      float wIdeal = std::sqrt((12.0*sigma*sigma/n)+1);  // Ideal averaging filter width
      int wl = std::floor(wIdeal);  if(wl%2==0) wl--;
      int wu = wl+2;

      float mIdeal = (12.0*sigma*sigma
                      - n*static_cast<float>(wl*wl)
                      - 4.0*n*static_cast<float>(wl)
                      - 3.0*n)/(-4.0*static_cast<float>(wl) - 4.0);

      int m = std::round(mIdeal);

      // var sigmaActual = Math.sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 );
      for(int i=0; i<box_number; i++)
        boxes.push_back(static_cast<float>(i<m?wl:wu));
    }
}

namespace dehancer {

    class ConvolveLineKernel: public Kernel {
    public:
        ConvolveLineKernel(const void* command_queue,
                           const Texture& s,
                           const Texture& d,
                           const dehancer::math::KernelLine& kernel_line
        ):
                dehancer::Kernel(command_queue,"convolve_line_kernel", s, d),
                kernel_line_(kernel_line)
        {
          mem_kernel_line_ = MemoryHolder::Make(get_command_queue(),
                                                kernel_line_.first.data(),
                                                kernel_line_.first.size()*sizeof(float));
          mem_kernel_sum_ = MemoryHolder::Make(get_command_queue(),
                                               kernel_line_.second.data(),
                                               kernel_line_.second.size()*sizeof(float));
        };

        void setup(CommandEncoder &encode) override {
          encode.set(&kernel_size_,sizeof(kernel_size_),2);
          encode.set(mem_kernel_line_,3);
          encode.set(mem_kernel_sum_,4);
        }

    private:
        dehancer::math::KernelLine kernel_line_;
        int    kernel_size_;
        Memory mem_kernel_line_;
        Memory mem_kernel_sum_;
    };

    class Channels2Texture: public Kernel {
    public:
        Channels2Texture(const void *command_queue,
                         const Memory& reds,
                         const Memory& greens,
                         const Memory& blues,
                         const Memory& alphas,
                         const Texture& d,
                         bool wait_until_completed=WAIT_UNTIL_COMPLETED):
                Kernel(command_queue, "channels_to_image", nullptr, d, wait_until_completed),
                reds_(reds),
                greens_(greens),
                blues_(blues),
                alphas_(alphas)
        {

        }

        void setup(CommandEncoder &encode) override {
          encode.set(reds_,1);
          encode.set(greens_,2);
          encode.set(blues_,3);
          encode.set(alphas_,4);
        }

    private:
        Memory reds_;
        Memory greens_;
        Memory blues_;
        Memory alphas_;
    };

    class BoxBlur: public Function {
    public:
        BoxBlur(const void *command_queue,
                const Memory& channel_in,
                const Memory& channel_out,
                size_t w,
                size_t h,
                int radius,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED):
                Function(command_queue, "box_blur_swap_kernel", wait_until_completed),
                channel_in_(channel_in),
                channel_out_(channel_out),
                w_(w),
                h_(h),
                radius_(radius),
                box_blur_horizontal_kernel_(new Function(command_queue, "box_blur_horizontal_kernel")),
                box_blur_vertical_kernel_(new Function(command_queue, "box_blur_vertical_kernel"))
        {

        }

        void process(){

          execute([this](CommandEncoder& command){
              command.set(channel_in_,0);
              command.set(channel_out_,1);
              int w = w_, h = h_;
              command.set(&w,sizeof(w),2);
              command.set(&h,sizeof(h),3);
              command.set(&radius_,sizeof(radius_),4);
              return (CommandEncoder::Size){this->w_,this->h_,1};
          });

          box_blur_horizontal_kernel_->execute([this](CommandEncoder& command){
              command.set(this->channel_out_,0);
              command.set(this->channel_in_,1);
              int w = this->w_, h = this->h_;
              command.set(&w,sizeof(w),2);
              command.set(&h,sizeof(h),3);
              command.set(&this->radius_,sizeof(this->radius_),4);
              return (CommandEncoder::Size){this->w_,this->h_,1};
          });

          box_blur_vertical_kernel_->execute([this](CommandEncoder& command){
              command.set(this->channel_in_,0);
              command.set(this->channel_out_,1);
              int w = this->w_, h = this->h_;
              command.set(&w,sizeof(w),2);
              command.set(&h,sizeof(h),3);
              command.set(&this->radius_,sizeof(this->radius_),4);
              return (CommandEncoder::Size){this->w_,this->h_,1};
          });
        }

    private:
        const Memory& channel_in_;
        const Memory& channel_out_;
        size_t w_;
        size_t h_;
        int radius_;
        std::shared_ptr<Function> box_blur_horizontal_kernel_;
        std::shared_ptr<Function> box_blur_vertical_kernel_;
    };

    class GaussianBlur: public Kernel {
    public:

        enum class Direction:int {
            horizontal = 0,
            vertical
        };

        GaussianBlur(const void* command_queue,
                     const Texture& s,
                     const Texture& d,
                     int radius,
                //Direction direction
                     bool wait_until_completed = WAIT_UNTIL_COMPLETED
        ):
                dehancer::Kernel(command_queue,
                        //direction == Direction::horizontal
                        //? "box_blur_horizontal_kernel"
                        //: "box_blur_vertical_kernel",
                                 "image_to_channels",
                                 s,
                                 d,
                                 wait_until_completed),
                radius_(radius),
                w_(s->get_width()),
                h_(s->get_height()),
                size_(w_*h_*sizeof(float ))
                ,
                reds_(MemoryHolder::Make(get_command_queue(),size_)),
                greens_(MemoryHolder::Make(get_command_queue(),size_)),
                blues_(MemoryHolder::Make(get_command_queue(),size_)),
                alphas_(MemoryHolder::Make(get_command_queue(),size_)),
                reds_out_(MemoryHolder::Make(get_command_queue(),size_)),
                greens_out_(MemoryHolder::Make(get_command_queue(),size_)),
                blues_out_(MemoryHolder::Make(get_command_queue(),size_)),
                alphas_out_(MemoryHolder::Make(get_command_queue(),size_)),
                channels_2_texture_(command_queue,reds_out_,greens_,blues_,alphas_,d)
                //,
                //box_blur_horizontal_kernel_(new Function(command_queue, "box_blur_horizontal_kernel")),
                //box_blur_vertical_kernel_(new Function(command_queue, "box_blur_vertical_kernel"))
        {
        }

//        void setup(CommandEncoder &command) override {
//          //encode.set(&radius_,sizeof(radius_),2);
//          command.set(this->get_source(),0);
//          command.set(this->reds_,1);
//          command.set(this->greens_,2);
//          command.set(this->blues_,3);
//          command.set(this->alphas_,4);
//        }

        void process() override {

          execute([this](CommandEncoder& command){
              //this->setup(command);
              command.set(this->get_source(),0);
              //command.set(this->get_destination(),1);
              //command.set(&radius_, sizeof(radius_), 2);
              command.set(this->reds_,1);
              command.set(this->greens_,2);
              command.set(this->blues_,3);
              command.set(this->alphas_,4);
              return CommandEncoder::Size::From(this->get_source());
          });

          std::vector<float> k_boxes;
          dehancer::math::make_gauss_boxes(k_boxes,radius_,3);

          BoxBlur(get_command_queue(),reds_,reds_out_,w_,h_,(k_boxes[0]-1)/2).process();
          BoxBlur(get_command_queue(),reds_out_,reds_,w_,h_,(k_boxes[1]-1)/2).process();
          BoxBlur(get_command_queue(),reds_,reds_out_,w_,h_,(k_boxes[2]-1)/2).process();

          BoxBlur(get_command_queue(),greens_,greens_out_,w_,h_,(k_boxes[0]-1)/2).process();
          BoxBlur(get_command_queue(),greens_out_,greens_,w_,h_,(k_boxes[1]-1)/2).process();
          BoxBlur(get_command_queue(),greens_,greens_out_,w_,h_,(k_boxes[2]-1)/2).process();

          BoxBlur(get_command_queue(),blues_,blues_out_,w_,h_,(k_boxes[0]-1)/2).process();
          BoxBlur(get_command_queue(),blues_out_,blues_,w_,h_,(k_boxes[1]-1)/2).process();
          BoxBlur(get_command_queue(),blues_,blues_out_,w_,h_,(k_boxes[2]-1)/2).process();

//          box_blur_horizontal_kernel_->execute([this](CommandEncoder& command){
//              command.set(this->reds_,0);
//              command.set(this->reds_out_,1);
//              int w = this->w_, h = this->h_;
//              command.set(&w,sizeof(w),2);
//              command.set(&h,sizeof(h),3);
//              command.set(&this->radius_,sizeof(this->radius_),4);
//              return (CommandEncoder::Size){this->w_,this->h_,1};
//          });
//
//          box_blur_vertical_kernel_->execute([this](CommandEncoder& command){
//              command.set(this->reds_out_,0);
//              command.set(this->reds_,1);
//              int w = this->w_, h = this->h_;
//              command.set(&w,sizeof(w),2);
//              command.set(&h,sizeof(h),3);
//              command.set(&this->radius_,sizeof(this->radius_),4);
//              return (CommandEncoder::Size){this->w_,this->h_,1};
//          });

          channels_2_texture_.process();
        }

    private:
        int radius_;
        size_t w_;
        size_t h_;
        size_t size_;
        Memory reds_;
        Memory greens_;
        Memory blues_;
        Memory alphas_;
        Memory reds_out_;
        Memory greens_out_;
        Memory blues_out_;
        Memory alphas_out_;
        Channels2Texture channels_2_texture_;
        //std::shared_ptr<Function> box_blur_horizontal_kernel_;
        //std::shared_ptr<Function> box_blur_vertical_kernel_;
    };


}

int run_bench(int num, const void* device, std::string patform) {

//  auto k_line = dehancer::math::KernelLine();
//
//  dehancer::math::make_gaussian_kernel(k_line, 20, 0.001, 16);
//
//  int i = 0;
//  std::cerr << "precompiled kernel: " << std::endl;
//  for (auto f: k_line.first) {
//    std::cerr << " ... ... ["<<i<<"] = " << f << " / " << k_line.second[i] << std::endl;
//    i ++;
//  }
//
//  std::vector<float> k_boxes;
//  dehancer::math::make_gauss_boxes(k_boxes,0.1,3);
//
//  i = 0;
//  std::cerr << "blur boxes: " << std::endl;
//  for (auto f: k_boxes) {
//    std::cerr << " ... ... ["<<i<<"] = " << f  << std::endl;
//    i ++;
//  }

  dehancer::TextureIO::Options::Type type = dehancer::TextureIO::Options::Type::png;
  std::string ext = dehancer::TextureIO::extention_for(type);
  float compression = 0.3f;

  size_t width = 1920, height = 1080 ;

  auto command_queue = dehancer::DeviceCache::Instance().get_command_queue(dehancer::device::get_id(device));

  auto grid_kernel = dehancer::Function(command_queue, "grid_kernel", true);
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

      command_encoder.set(&levels, sizeof(levels), 0);
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

//  auto output_text2 = dehancer::TextureOutput(command_queue, width, height, nullptr, {
//          .type = type,
//          .compression = compression
//  });

  auto box_blur_line_kernel = dehancer::GaussianBlur(command_queue,
                                                     grid_text,
                                                     output_text.get_texture(),
                                                     20,
                                                     true
  );

//  auto box_blur_line_kernel2 = dehancer::GaussianBlur(command_queue,
//                                                     output_text.get_texture(),
//                                                     output_text2.get_texture(),
//                                                     10
//  );

  std::cout << "[convolve_line_kernel kernel " << grid_kernel.get_name() << " args: " << std::endl;
  for (auto &a: box_blur_line_kernel.get_arg_list()) {
    std::cout << std::setw(20) << a.name << "[" << a.index << "]: " << a.type_name << std::endl;
  }

  std::chrono::time_point<std::chrono::system_clock> clock_begin
          = std::chrono::system_clock::now();

  box_blur_line_kernel.process();
  //box_blur_line_kernel2.process();

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

  std::cout << "[convolve-line "
            <<patform<<"/"<<device_type_str
            <<" ("
            <<dehancer::device::get_name(device)
            <<")]:\t" << seconds.count() << "s "
            << ", for a " << width << "x" << height << " pixels" << std::endl;


  std::string out_file_result = "convolve-line-"+patform+"-result-"; out_file_result.append(std::to_string(num)); out_file_result.append(ext);
  {
    std::ofstream result_os(out_file_result, std::ostream::binary | std::ostream::trunc);
    result_os << output_text;
  }

  return 0;
}

void test_bench(std::string platform) {
  try {
    auto devices = dehancer::DeviceCache::Instance().get_device_list();
    assert(!devices.empty());

    int dev_num = 0;
    std::cout << "Platform: " << platform << std::endl;
    for (auto d: devices) {
      std::cout << " #" << dev_num++ << std::endl;
      std::cout << "    Device '" << dehancer::device::get_name(d) << " ["<<dehancer::device::get_id(d)<<"]'"<< std::endl;
    }

    std::cout << "Bench: " << std::endl;
    dev_num = 0;

    for (auto d: devices) {
//#if __APPLE__
//      if (dehancer::device::get_type(d) == dehancer::device::Type::cpu) continue;
//#endif
      if (run_bench(dev_num++, d, platform)!=0) return;
    }

    //if (run_bench(1, devices[1], platform)!=0) return;

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}