//
// Created by denn nevera on 01/12/2020.
//

#include "dehancer/gpu/operations/GaussianBlur.h"

#include <cmath>
#include "dehancer/gpu/math/ConvolveUtils.h"

namespace dehancer {
    
    struct GaussianBlurOptions {
        std::array<float, 4> radius_array;
        float  accuracy;
    };
    
    auto kernel_blur = [](int index, std::vector<float>& data, const std::optional<std::any>& user_data) {
        
        
        data.clear();
        
        if (!user_data.has_value()) return ;
        
        auto options = std::any_cast<GaussianBlurOptions>(user_data.value());
        
        auto radius = options.radius_array.at(index);
        
        if (radius==0) return ;
        
        float sigma = radius/2.0f;
        int kRadius = (int)std::ceil(sigma*std::sqrt(-2.0f*std::log(options.accuracy)))+1;
        int maxRadius = (int)std::ceil(radius/2+1) * 4 - 1;
        
        kRadius = std::max(kRadius,maxRadius);
        
        auto size = kRadius;
        if (size%2==0) size+=1;
        if (size<3) size=3;
        
        dehancer::math::make_gaussian_kernel(data, size, radius/2.0f);
    };
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                const Texture &s,
                                const Texture &d,
                                std::array<float, 4> radius,
                                const ChannelDesc::Transform& transform,
                                DHCR_EdgeMode    edge_mode,
                                float             accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path):
            UnaryKernel(command_queue,s,d,{
                                .row = kernel_blur,
                                .col = kernel_blur,
                                .user_data = (GaussianBlurOptions){radius,accuracy_},
                                .edge_mode = edge_mode
                        },
                        transform,
                        wait_until_completed,
                        library_path)
    {
    }
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                const Texture &s,
                                const Texture &d,
                                float radius,
                                const ChannelDesc::Transform& transform,
                                DHCR_EdgeMode edge_mode,
                                float accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path):
            GaussianBlur(command_queue,s,d,
                         {radius,radius,radius,0},
                         transform,
                         edge_mode, accuracy_,
                         wait_until_completed,
                         library_path) {
      
    }
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                std::array<float, 4> radius,
                                const ChannelDesc::Transform& transform,
                                DHCR_EdgeMode edge_mode,
                                float accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path):
            GaussianBlur(command_queue, nullptr, nullptr,
                         radius, transform, edge_mode, accuracy_, wait_until_completed, library_path){
      
    }
    
    GaussianBlur::GaussianBlur (const void *command_queue,
                                float radius,
                                const ChannelDesc::Transform& transform,
                                DHCR_EdgeMode edge_mode,
                                float accuracy_,
                                bool wait_until_completed,
                                const std::string &library_path)
            :GaussianBlur(command_queue,
                          {radius,radius,radius,0},
                          transform,
                          edge_mode,accuracy_,wait_until_completed,library_path) {
      
    }
    
    void GaussianBlur::set_radius (float radius) {
      set_radius({radius,radius,radius,0});
    }
    
    void GaussianBlur::set_radius (std::array<float, 4> radius) {
      auto options = get_options();
      auto data = options.user_data.has_value()
                  ? std::any_cast<GaussianBlurOptions>(options.user_data.value())
                  : (GaussianBlurOptions){radius,GaussianBlur::accuracy};
      data.radius_array = radius;
      set_user_data(data);
    }
    
    void GaussianBlur::set_accuracy (float accuracy_) {
      auto options = get_options();
      auto data = options.user_data.has_value()
                  ? std::any_cast<GaussianBlurOptions>(options.user_data.value())
                  : (GaussianBlurOptions){{0,0,0,0},accuracy_};
      data.accuracy = accuracy_;
      set_user_data(data);
    }
    
    std::array<float, 4> GaussianBlur::get_radius () const {
      auto options = get_options();
      if (options.user_data.has_value())
        return std::any_cast<GaussianBlurOptions>(options.user_data.value()).radius_array;
      else
        return {};
    }
    
    //
    // experimental blur as boxed blur
    //

//    namespace _blur_impl_ {
//        class SwapChannels: public Function {
//        public:
//            SwapChannels(const void *command_queue,
//                         const Memory& channel_in,
//                         const Memory& channel_out,
//                         size_t w,
//                         size_t h,
//                         bool wait_until_completed = WAIT_UNTIL_COMPLETED,
//                         const std::string& library_path = ""):
//                    Function(command_queue, "swap_channels_kernel", wait_until_completed, library_path),
//                    channel_in_(channel_in),
//                    channel_out_(channel_out),
//                    w_(w),
//                    h_(h){}
//
//            void process() {
//              execute([this](CommandEncoder &command) {
//                  command.set(channel_in_, 0);
//                  command.set(channel_out_, 1);
//                  int w = w_, h = h_;
//                  command.set(&w, sizeof(w), 2);
//                  command.set(&h, sizeof(h), 3);
//                  return (CommandEncoder::Size) {this->w_, this->h_, 1};
//              });
//            }
//
//        protected:
//            const Memory& channel_in_;
//            const Memory& channel_out_;
//            size_t w_;
//            size_t h_;
//        };
//
//        class BoxBlurChannels: public SwapChannels {
//        public:
//            BoxBlurChannels(const void *command_queue,
//                            const Memory& channel_in,
//                            const Memory& channel_out,
//                            size_t w,
//                            size_t h,
//                            int radius,
//                            bool wait_until_completed = WAIT_UNTIL_COMPLETED,
//                            const std::string& library_path = ""):
//                    SwapChannels(command_queue, channel_in, channel_out, w, h , wait_until_completed, library_path),
//                    radius_(radius),
//                    box_blur_horizontal_kernel_(new Function(command_queue, "box_blur_horizontal_kernel", "")),
//                    box_blur_vertical_kernel_(new Function(command_queue, "box_blur_vertical_kernel", ""))
//            {
//
//            }
//
//            void process() {
//
//              SwapChannels::process();
//
//              box_blur_horizontal_kernel_->execute([this](CommandEncoder& command){
//                  command.set(this->channel_out_,0);
//                  command.set(this->channel_in_,1);
//                  int w = this->w_, h = this->h_;
//                  command.set(&w,sizeof(w),2);
//                  command.set(&h,sizeof(h),3);
//                  command.set(&this->radius_,sizeof(this->radius_),4);
//                  return (CommandEncoder::Size){this->w_,this->h_,1};
//              });
//
//              box_blur_vertical_kernel_->execute([this](CommandEncoder& command){
//                  command.set(this->channel_in_,0);
//                  command.set(this->channel_out_,1);
//                  int w = this->w_, h = this->h_;
//                  command.set(&w,sizeof(w),2);
//                  command.set(&h,sizeof(h),3);
//                  command.set(&this->radius_,sizeof(this->radius_),4);
//                  return (CommandEncoder::Size){this->w_,this->h_,1};
//              });
//            }
//
//        private:
//            int radius_;
//            std::shared_ptr<Function> box_blur_horizontal_kernel_;
//            std::shared_ptr<Function> box_blur_vertical_kernel_;
//        };
//    }

//    GaussianBlur::GaussianBlur(const void* command_queue,
//                               const Texture& s,
//                               const Texture& d,
//                               std::array<int,4> radius,
//                               bool wait_until_completed,
//                               const std::string& library_path
//    ):
//            ChannelsInput (command_queue, s, wait_until_completed, library_path),
//            radius_(radius),
//            w_(s->get_width()),
//            h_(s->get_height()),
//            channels_out_(ChannelsHolder::Make(command_queue,s->get_width(),s->get_height())),
//            channels_finalizer_(command_queue, d, channels_out_, wait_until_completed)
//    {
//      for (int i = 0; i < radius_.size(); ++i) {
//        dehancer::math::make_gaussian_boxes(radius_boxes_[i], radius_[i]/2, box_number_);
//      }
//    }
//
//    void GaussianBlur::process() {
//
//      ChannelsInput::process();
//
//      for (int i = 0; i < channels_out_->size(); ++i) {
//        auto in = this->get_channels()->at(i);
//        auto out = channels_out_->at(i);
//        for (int j = 0; j < box_number_; ++j) {
//          auto r = (radius_boxes_[i][j]-1)/2;
//          if (r==0){
//            _blur_impl_::SwapChannels(get_command_queue(), in, out, w_, h_, r).process();
//            break;
//          }
//          _blur_impl_::BoxBlurChannels(get_command_queue(), in, out, w_, h_, r).process();
//          std::swap(in,out);
//        }
//      }
//
//      channels_finalizer_.process();
//    }
}