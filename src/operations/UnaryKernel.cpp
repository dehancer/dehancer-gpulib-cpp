//
// Created by denn on 09.01.2021.
//

#include "dehancer/gpu/operations/UnaryKernel.h"

#include <utility>

namespace dehancer {
    
    struct UnaryKernelImpl{
        ChannelsInput* root_;
        UnaryKernel::KernelFunction row_func;
        UnaryKernel::KernelFunction col_func;
        UnaryKernel::UserData       user_data;
        DHCR_EdgeMode    edge_mode;
        std::array<dehancer::Memory,4> row_weights;
        std::array<int,4> row_sizes{};
        std::array<dehancer::Memory,4> col_weights;
        std::array<int,4> col_sizes{};
        size_t width;
        size_t height;
        Channels channels_out;
        std::shared_ptr<ChannelsOutput> channels_finalizer;
        
        UnaryKernelImpl(
                ChannelsInput* root,
                const Texture& s,
                const Texture& d,
                UnaryKernel::KernelFunction row_func,
                UnaryKernel::KernelFunction col_func,
                UnaryKernel::UserData        user_data,
                DHCR_EdgeMode    address_mode
        );
    };
    
    UnaryKernelImpl::UnaryKernelImpl (ChannelsInput *root, const Texture &s, const Texture &d,
                                      UnaryKernel::KernelFunction row_func,
                                      UnaryKernel::KernelFunction col_func,
                                      UnaryKernel::UserData  user_data,
                                      DHCR_EdgeMode address_mode) :
            root_(root),
            row_func(std::move(row_func)),
            col_func(std::move(col_func)),
            user_data(std::move(user_data)),
            edge_mode(address_mode),
            width(s?s->get_width():0),
            height(s?s->get_height():0),
            channels_out(width>0?ChannelsHolder::Make(root_->get_command_queue(), width, height): nullptr),
            channels_finalizer(std::make_shared<ChannelsOutput>(root_->get_command_queue(), d, root_->get_channels(), root_->get_wait_completed()))
    {}
    
    UnaryKernel::UnaryKernel(const void* command_queue,
                             const Texture& s,
                             const Texture& d,
                             const Options& options,
                             bool wait_until_completed,
                             const std::string& library_path
    ):
            ChannelsInput (command_queue, s, wait_until_completed, library_path),
            impl_(std::make_shared<UnaryKernelImpl>(
                    this,
                    s,
                    d,
                    options.row,
                    options.col,
                    options.user_data,
                    options.edge_mode
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
              
              int a = impl_->edge_mode;
              command.set(a, 6);
              
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
              
              command.set(impl_->col_weights.at(0), 4);
              command.set(impl_->col_sizes[0], 5);
              
              int a = impl_->edge_mode;
              command.set(a, 6);
              
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
      impl_->channels_out = impl_->width>0?ChannelsHolder::Make(get_command_queue(), impl_->width, impl_->height): nullptr;
    }
    
    void UnaryKernel::set_destination (const Texture &dest) {
      dehancer::ChannelsInput::set_destination(nullptr);
      impl_->channels_finalizer = std::make_shared<ChannelsOutput>(
              get_command_queue(),
              dest,
              get_channels(),
              get_wait_completed());
    }
    
    UnaryKernel::UnaryKernel (const void *command_queue,
                              const UnaryKernel::Options &options,
                              bool wait_until_completed,
                              const std::string &library_path):
            UnaryKernel(command_queue, nullptr, nullptr, options, wait_until_completed, library_path)
    {
    }
    
    [[maybe_unused]] void UnaryKernel::set_edge_mode (DHCR_EdgeMode address) {
      impl_->edge_mode = address;
    }
    
    void UnaryKernel::set_user_data (const UserData &user_data) {
      impl_->user_data = user_data;
      recompute_kernel();
    }
    
    void UnaryKernel::set_options (const UnaryKernel::Options &options) {
      impl_->user_data = options.user_data ? options.user_data : impl_->user_data;
      impl_->row_func  = options.row ? options.row : impl_->row_func;
      impl_->col_func  = options.col ? options.col : impl_->col_func;
      impl_->edge_mode = options.edge_mode;
      recompute_kernel();
    }
    
    UnaryKernel::Options UnaryKernel::get_options () const {
      return {
        impl_->row_func,
        impl_->col_func,
        impl_->user_data,
        impl_->edge_mode
      };
    }
    
    void UnaryKernel::recompute_kernel () {
  
      if (!impl_->user_data.has_value()) return;
      
      for (int i = 0; i < 4; ++i) {
  
        std::vector<float> buf;
  
        if (impl_->row_func && impl_->user_data) {
          
          impl_->row_func(i, buf, impl_->user_data);
          impl_->row_sizes[i] = buf.size();
    
          if (buf.empty())
            impl_->row_weights[i] = nullptr;
          else
            impl_->row_weights[i] = dehancer::MemoryHolder::Make(get_command_queue(),
                                                                 buf.data(),
                                                                 buf.size() * sizeof(float));
        }
        if (impl_->col_func && impl_->user_data) {
          buf.clear();
    
          impl_->col_func(i, buf, impl_->user_data);
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
  
}