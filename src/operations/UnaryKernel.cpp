//
// Created by denn on 09.01.2021.
//

#include "dehancer/gpu/operations/UnaryKernel.h"

namespace dehancer {
    
    struct UnaryKernelImpl{
        ChannelsInput* root_;
        UnaryKernel::KernelFunction row_func;
        UnaryKernel::KernelFunction col_func;
        UnaryKernel::UserData       user_data;
        DHCR_EdgeAddress    address_mode;
        std::array<dehancer::Memory,4> row_weights;
        std::array<int,4> row_sizes{};
        std::array<dehancer::Memory,4> col_weights;
        std::array<int,4> col_sizes{};
        size_t width;
        size_t height;
        Channels channels_out;
        ChannelsOutput channels_finalizer;
        
        UnaryKernelImpl(
                ChannelsInput* root,
                const Texture& s,
                const Texture& d,
                const UnaryKernel::KernelFunction& row_func,
                const UnaryKernel::KernelFunction& col_func,
                const UnaryKernel::UserData&       user_data,
                DHCR_EdgeAddress    address_mode
        ):
                root_(root),
                row_func(row_func),
                col_func(col_func),
                user_data(user_data),
                address_mode(address_mode),
                width(s->get_width()),
                height(s->get_height()),
                channels_out(ChannelsHolder::Make(root_->get_command_queue(), width, height)),
                channels_finalizer(root_->get_command_queue(), d, root_->get_channels(), root_->get_wait_completed())
        {};
    };
    
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
                    options.address_mode
            ))
    {
      
      for (int i = 0; i < 4; ++i) {
        std::vector<float> buf;
        
        impl_->row_func(i, buf, impl_->user_data);
        impl_->row_sizes[i] = buf.size();
        
        if (buf.empty())
          impl_->row_weights[i] = nullptr;
        else
          impl_->row_weights[i] = dehancer::MemoryHolder::Make(get_command_queue(),
                                                               buf.data(),
                                                                buf.size() * sizeof(float));
        
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
              
              int a = impl_->address_mode;
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
    
              int a = impl_->address_mode;
              command.set(a, 6);
              
              return (CommandEncoder::Size) {impl_->width, impl_->height, 1};
          });
        }
      }
      
      impl_->channels_finalizer.process();
    }
    
}