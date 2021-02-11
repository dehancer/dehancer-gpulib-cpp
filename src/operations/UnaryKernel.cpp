//
// Created by denn on 09.01.2021.
//

#include "dehancer/gpu/operations/UnaryKernel.h"

#include <utility>

namespace dehancer {
    
    struct UnaryKernelImpl{
        UnaryKernel* root_;
        UnaryKernel::Options options_;
        ChannelDesc::Transform transform_ {};
        std::array<dehancer::Memory,4> row_weights;
        std::array<int,4> row_sizes{};
        std::array<dehancer::Memory,4> col_weights;
        std::array<int,4> col_sizes{};
        size_t width;
        size_t height;
        Channels channels_out;
        std::shared_ptr<ChannelsOutput> channels_finalizer;
        bool has_mask_;
        Texture mask_;
        
        UnaryKernelImpl(
                UnaryKernel* root,
                const Texture& s,
                const Texture& d,
                const UnaryKernel::Options& options,
                const ChannelDesc::Transform& transform
        );
        
        [[nodiscard]] ChannelDesc::Transform get_output_transform() const {
          ChannelDesc::Transform t = transform_;
          if (t.direction == ChannelDesc::TransformDirection::forward) {
            t.direction = ChannelDesc::TransformDirection::inverse;
          }
          else if (t.direction == ChannelDesc::TransformDirection::inverse) {
            t.direction = ChannelDesc::TransformDirection::forward;
          }
          return t;
        }
    };
    
    UnaryKernelImpl::UnaryKernelImpl (UnaryKernel *root,
                                      const Texture &s,
                                      const Texture &d,
                                      const UnaryKernel::Options& options,
                                      const ChannelDesc::Transform& transform
    ) :
            root_(root),
            options_(options),
            transform_(transform),
            width(s?s->get_width():0),
            height(s?s->get_height():0),
            channels_out(ChannelsHolder::Make(
                    root_->get_command_queue(),
                    root_->get_channels()->get_desc()
                    //(ChannelDesc){
                    //  .width = width,
                    //  .height = height
                    //}
            )),
            channels_finalizer(std::make_shared<ChannelsOutput>(
                    root_->get_command_queue(),
                    d,
                    root_->get_channels(),
                    get_output_transform(),
                    root_->get_wait_completed())
            )
    {
      has_mask_ = options_.mask != nullptr;
      if (!has_mask_) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(root->get_command_queue(),mem);
      }
      else
        mask_ = options_.mask;
    }
    
    UnaryKernel::UnaryKernel(const void* command_queue,
                             const Texture& s,
                             const Texture& d,
                             const Options& options,
                             const ChannelDesc::Transform& transform,
                             bool wait_until_completed,
                             const std::string& library_path
    ):
            ChannelsInput (command_queue,
                           s,
                           transform,
                           {1.0f,1.0f,1.0f,1.0f},
                           wait_until_completed,
                           library_path),
            impl_(std::make_shared<UnaryKernelImpl>(
                    this,
                    s,
                    d,
                    options,
                    transform
            ))
    {
      recompute_kernel();
    }
    
    void UnaryKernel::process() {
      
      ChannelsInput::process();
      
      auto horizontal_kernel = Function(get_command_queue(),
                                        "kernel_convolve_horizontal");
      
      auto vertical_kernel = Function(get_command_queue(),
                                      "kernel_convolve_vertical",
                                      get_wait_completed());
      
      for (int i = 0; i < get_channels()->size(); ++i) {
        
        if (impl_->row_weights[i]) {
          
          horizontal_kernel.execute([this, i] (CommandEncoder &command) {
              auto in = get_channels()->at(i);
              auto out = impl_->channels_out->at(i);
              
              command.set(in, 0);
              command.set(out, 1);
              
              int w = impl_->width, h = impl_->height;
              command.set(w, 2);
              command.set(h, 3);
              
              command.set(impl_->row_weights.at(i), 4);
              command.set(impl_->row_sizes[i], 5);
              
              int a = impl_->options_.edge_mode;
              command.set(a, 6);
              
              command.set(impl_->has_mask_, 7);
              command.set(impl_->mask_, 8);
              command.set(i, 9);
              
              return (CommandEncoder::Size) {impl_->width, impl_->height, 1};
          });
        }
        
        if (impl_->col_weights[i]) {
          vertical_kernel.execute([this, i](CommandEncoder &command) {
              auto in = impl_->channels_out->at(i);
              auto out = get_channels()->at(i);
              
              command.set(in, 0);
              command.set(out, 1);
              
              int w = impl_->width, h = impl_->height;
              command.set(w, 2);
              command.set(h, 3);
              
              command.set(impl_->col_weights.at(i), 4);
              command.set(impl_->col_sizes[i], 5);
              
              int a = impl_->options_.edge_mode;
              command.set(a, 6);
              
              command.set(impl_->has_mask_, 7);
              command.set(impl_->mask_, 8);
              command.set(i, 9);
              
              return (CommandEncoder::Size) {impl_->width, impl_->height, 1};
          });
        }
      }
      
      impl_->channels_finalizer->process();
    }
    
    void UnaryKernel::set_source (const Texture &s) {
      dehancer::ChannelsInput::set_source(s);
      impl_->width = s?s->get_width():0;
      impl_->height = s?s->get_height():0;
      if (impl_->channels_out){
        //if (impl_->channels_out->get_height()!=impl_->height || impl_->channels_out->get_width()!=impl_->width)
        // impl_->channels_out = nullptr;
        for (int i = 0; i < impl_->channels_out->size(); ++i) {
          if (impl_->channels_out->get_height(i)!=get_channels()->get_height(i)
              ||
              impl_->channels_out->get_width(i)!=get_channels()->get_width(i)) {
            impl_->channels_out = nullptr;
            break;
          }
        }
      }
      if (!impl_->channels_out)
        impl_->channels_out = get_channels()->get_desc().make(get_command_queue()); //ChannelsHolder::Make(get_command_queue(), impl_->width, impl_->height);
    }
    
    void UnaryKernel::set_destination (const Texture &dest) {
      dehancer::ChannelsInput::set_destination(nullptr);
      impl_->channels_finalizer->set_destination(dest);
      impl_->channels_finalizer->set_channels(get_channels());
    }
    
    UnaryKernel::UnaryKernel (const void *command_queue,
                              const UnaryKernel::Options &options,
                              const ChannelDesc::Transform& transform,
                              bool wait_until_completed,
                              const std::string &library_path):
            UnaryKernel(command_queue, nullptr, nullptr, options, transform, wait_until_completed, library_path)
    {
    }
    
    [[maybe_unused]] void UnaryKernel::set_edge_mode (DHCR_EdgeMode address) {
      impl_->options_.edge_mode = address;
    }
    
    void UnaryKernel::set_user_data (const UserData &user_data) {
      impl_->options_.user_data = user_data;
      recompute_kernel();
    }
    
    void UnaryKernel::set_transform (const ChannelDesc::Transform &transform) {
      impl_->transform_ = transform;
      ChannelsInput::set_transform(transform);
      impl_->channels_finalizer->set_transform(impl_->get_output_transform());
      recompute_kernel();
    }
    
    void UnaryKernel::set_options (const UnaryKernel::Options &options) {
      impl_->options_ = options;
      impl_->has_mask_ = impl_->options_.mask != nullptr;
      if (!impl_->has_mask_) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        impl_->mask_ = desc.make(get_command_queue(),mem);
      }
      else
        impl_->mask_ = impl_->options_.mask;
      recompute_kernel();
    }
    
    void UnaryKernel::set_mask(const Texture &mask) {
      impl_->options_.mask = mask;
      impl_->has_mask_ = impl_->options_.mask != nullptr;
      if (!impl_->has_mask_) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        impl_->mask_ = desc.make(get_command_queue(),mem);
      }
      else
        impl_->mask_ = impl_->options_.mask;
    }
    
    const UnaryKernel::Options& UnaryKernel::get_options () const {
      return impl_->options_;
    }
    
    UnaryKernel::Options &UnaryKernel::get_options () {
      return impl_->options_;
    }
    
    void UnaryKernel::recompute_kernel () {
      
      if (!impl_->options_.user_data.has_value()) return;
      
      for (int i = 0; i < get_channels()->size(); ++i) {
        
        std::vector<float> buf;
        
        if (impl_->options_.row && impl_->options_.user_data) {
          
          impl_->options_.row(i, buf, impl_->options_.user_data);
          impl_->row_sizes[i] = buf.size();
          
          if (buf.empty())
            impl_->row_weights[i] = nullptr;
          else
            impl_->row_weights[i] = dehancer::MemoryHolder::Make(get_command_queue(),
                                                                 buf.data(),
                                                                 buf.size() * sizeof(float));
        }
        if (impl_->options_.col && impl_->options_.user_data) {
          buf.clear();
          
          impl_->options_.col(i, buf, impl_->options_.user_data);
          impl_->col_sizes[i] = buf.size();
          
          if (buf.empty())
            impl_->col_weights[i] = nullptr;
          else
            impl_->col_weights[i] = dehancer::MemoryHolder::Make(get_command_queue(),
                                                                 buf.data(),
                                                                 buf.size() * sizeof(float));
        }
      }
    }
    
    const ChannelDesc::Transform &UnaryKernel::get_transform () const {
      return impl_->transform_;
    }
    
    void UnaryKernel::process (const Texture &source, const Texture &destination) {
      ChannelsInput::process(source, destination);
    }
  
}