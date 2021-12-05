//
// Created by denn on 09.01.2021.
//

#include "dehancer/gpu/operations/UnaryKernel.h"
#include "dehancer/gpu/Log.h"
#include <utility>

#define __TEST_NOT_SKIP__ 1

namespace dehancer {
    
    struct UnaryKernelImpl{
        
        UnaryKernel* root_;
        UnaryKernel::Options options_;
        ChannelsDesc::Transform transform_ {};
        std::array<dehancer::Memory,4> row_weights;
        
        std::array<int,4> row_sizes{};
        std::array<int,4> col_sizes{};
        
        std::array<dehancer::Memory,4> col_weights;
        std::shared_ptr<ChannelsInput>  channels_transformer;
        std::shared_ptr<ChannelsOutput> channels_finalizer;
        Channels channels_unary_ops;
        ChannelsDesc::Scale2D channels_scale;
        
        ChannelsDesc::ActiveChannelsMask amask;
        
        bool has_mask_;
        Texture mask_;
        
        std::string library_path;
        
        std::shared_ptr<Function> horizontal_kernel;
        std::shared_ptr<Function> vertical_kernel;
        
        UnaryKernelImpl(
                UnaryKernel* root,
                const Texture& s,
                const Texture& d,
                const UnaryKernel::Options& options,
                const ChannelsDesc::Transform& transform,
                const std::string& library_path
        );
        
        [[nodiscard]] ChannelsDesc::Transform get_output_transform() const {
          ChannelsDesc::Transform t = transform_;
          
          if (t.direction == ChannelsDesc::TransformDirection::forward) {
            t.direction = ChannelsDesc::TransformDirection::inverse;
          }
          else if (t.direction == ChannelsDesc::TransformDirection::inverse) {
            t.direction = ChannelsDesc::TransformDirection::forward;
          }
          else
            t.direction = ChannelsDesc::TransformDirection::none;
          
          return t;
        }
        
        void recompute_kernel();
        
        void set_source(const Texture& source);
        
        void set_destination(const Texture& destination);
        
        void reset_unary_ops();
    };
    
    UnaryKernelImpl::UnaryKernelImpl (UnaryKernel *root,
                                      const Texture &s,
                                      const Texture &d,
                                      const UnaryKernel::Options& options,
                                      const ChannelsDesc::Transform& transform,
                                      const std::string& library_path
    ) :
            root_(root),
            options_(options),
            transform_(transform),
            channels_transformer(nullptr),
            channels_finalizer(nullptr),
            channels_unary_ops(nullptr),
            channels_scale({1.f,1.f,1.f,1.0f}),
            library_path(library_path),
            horizontal_kernel(std::make_shared<Function>(root_->get_command_queue(),
                                                         "kernel_convolve_horizontal",
                                                         root_->get_wait_completed(),
                                                         library_path)),
            vertical_kernel(std::make_shared<Function>(root_->get_command_queue(),
                                                       "kernel_convolve_vertical",
                                                       root_->get_wait_completed(),
                                                       library_path))
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
    
    
    void UnaryKernelImpl::recompute_kernel () {
      
      if (!options_.user_data.has_value()) return;
      
      for (int i = 0; i < 4; ++i) {
        
        std::vector<float> buf;
        
        if (options_.row && options_.user_data) {
          
          float scale = options_.row(i, buf, options_.user_data);
          
          row_sizes[i] = buf.size();
          
          if (buf.empty()) {
            row_weights[i] = nullptr;
            channels_scale.at(i).x = 1.0f;
            amask.at(i) = false;
          }
          else {
            for (auto& d: buf ) d *= options_.amplify;
            row_weights[i] = dehancer::MemoryHolder::Make(root_->get_command_queue(),
                                                          buf.data(),
                                                          buf.size() * sizeof(float));
            channels_scale.at(i).x = scale;
            amask.at(i) = true;
          }
        }
        if (options_.col && options_.user_data) {
          buf.clear();
          
          float scale = options_.col(i, buf, options_.user_data);
          col_sizes[i] = buf.size();
          
          if (buf.empty()) {
            col_weights[i] = nullptr;
            channels_scale.at(i).y = 1.0f;
          }
          else {
            for (auto& d: buf ) d *= options_.amplify;
            col_weights[i] = dehancer::MemoryHolder::Make(root_->get_command_queue(),
                                                          buf.data(),
                                                          buf.size() * sizeof(float));
            channels_scale.at(i).y = scale;
          }
        }
      }
    }
    
    void UnaryKernelImpl::reset_unary_ops () {
      
      if (!channels_transformer || !channels_transformer->get_channels()) return;
      
      if (channels_transformer->get_channels())
        channels_unary_ops = channels_transformer->get_channels()->get_desc().make(root_->get_command_queue(), amask);
      
    }
    
    void UnaryKernelImpl::set_source (const Texture &source) {
      
      if (!source) return;
      
      channels_transformer = std::make_shared<ChannelsInput>(
              root_->get_command_queue(),
              source,
              transform_,
              channels_scale,
              amask,
              root_->get_wait_completed(),
              library_path
      );
      
    }
    
    void UnaryKernelImpl::set_destination (const Texture& destination) {
  
      if (!channels_transformer) return;
  
      channels_finalizer = std::make_shared<ChannelsOutput>(
              root_->get_command_queue(),
              destination,
              channels_transformer->get_channels(),
              get_output_transform(),
              root_->get_wait_completed(),
              library_path
      );
    }
    
    
    /***
     *
     * @param command_queue
     * @param s
     * @param d
     * @param options
     * @param transform
     * @param wait_until_completed
     * @param library_path
     */
    UnaryKernel::UnaryKernel(const void* command_queue,
                             const Texture& s,
                             const Texture& d,
                             const Options& options,
                             const ChannelsDesc::Transform& transform,
                             bool wait_until_completed,
                             const std::string& library_path
    ):
            PassKernel(command_queue, s, d, wait_until_completed, library_path),
            impl_(std::make_shared<UnaryKernelImpl>(
                    this,
                    s,
                    d,
                    options,
                    transform,
                    library_path
            ))
    {
      impl_->recompute_kernel();
    }
    
    void UnaryKernel::process() {
      
      impl_->set_source(get_source());
      impl_->set_destination(get_destination());
      
      if(!impl_->channels_transformer) return;
      if(!impl_->channels_finalizer) return;
      
      impl_->channels_transformer->process();
  
      impl_->channels_finalizer->set_source(get_source());
      
      impl_->reset_unary_ops();
      
      const ChannelsDesc::Transform& real_transform = impl_->transform_;
      
      impl_->recompute_kernel();
      if (impl_->channels_transformer)
        impl_->channels_transformer->set_transform(real_transform);
      
      if (impl_->channels_finalizer)
        impl_->channels_finalizer->set_transform(impl_->get_output_transform());
      
      #if  __TEST_NOT_SKIP__ == 1
      
      for (int i = 0; i < impl_->channels_transformer->get_channels()->size(); ++i) {
        
        if (impl_->row_weights[i]) {
          
          auto in = impl_->channels_transformer->get_channels()->at(i);
          auto out = impl_->channels_unary_ops->at(i);
          
          if (!in || !out) continue;
          
          impl_->horizontal_kernel->execute([this, i, &in, &out] (CommandEncoder &command) {
              
              command.set(in, 0);
              command.set(out, 1);
              
              int
                      w = (int)impl_->channels_unary_ops->get_width(i),
                      h = (int)impl_->channels_unary_ops->get_height(i);
              
              command.set(w, 2);
              command.set(h, 3);
              
              command.set(impl_->row_weights.at(i), 4);
              command.set(impl_->row_sizes[i], 5);
              
              int a = impl_->options_.edge_mode;
              command.set(a, 6);
              
              command.set(impl_->has_mask_, 7);
              command.set(impl_->mask_, 8);
              command.set(i, 9);
              
              CommandEncoder::Size size = {
                      .width = (size_t)w,
                      .height = (size_t)h,
                      .depth = 1
              };
              
              return size;
          });
        }
        
        if (impl_->col_weights[i]) {
          
          auto in = impl_->channels_unary_ops->at(i);
          auto out = impl_->channels_transformer->get_channels()->at(i);
          
          if (!in || !out) continue;
          
          impl_->vertical_kernel->execute([this, i, &in, &out](CommandEncoder &command) {
              
              command.set(in, 0);
              command.set(out, 1);
              
              int
                      w = (int)impl_->channels_transformer->get_channels()->get_width(i),
                      h = (int)impl_->channels_transformer->get_channels()->get_height(i);
              
              
              command.set(w, 2);
              command.set(h, 3);
              
              command.set(impl_->col_weights.at(i), 4);
              command.set(impl_->col_sizes[i], 5);
              
              int a = impl_->options_.edge_mode;
              command.set(a, 6);
              
              command.set(impl_->has_mask_, 7);
              command.set(impl_->mask_, 8);
              command.set(i, 9);
              
              CommandEncoder::Size size = {
                      .width = (size_t)w,
                      .height = (size_t)h,
                      .depth = 1
              };
              
              return size;
          });
        }
      }
      
      #endif
      
      impl_->channels_unary_ops = nullptr;
      
      impl_->set_destination(get_destination());
      impl_->channels_finalizer->process();
      
      impl_->channels_transformer = nullptr;
      impl_->channels_finalizer = nullptr;
    }
    
    void UnaryKernel::set_source (const Texture &s) {
      Kernel::set_source(s);
    }
    
    void UnaryKernel::set_destination (const Texture &dest) {
      Kernel::set_destination(dest);
    }
    
    UnaryKernel::UnaryKernel (const void *command_queue,
                              const UnaryKernel::Options &options,
                              const ChannelsDesc::Transform& transform,
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
      impl_->recompute_kernel();
    }
    
    void UnaryKernel::set_transform (const ChannelsDesc::Transform &transform) {
      impl_->transform_ = transform;
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
      
      impl_->recompute_kernel();
    }
    
    void UnaryKernel::set_amplify (float amplify) {
      impl_->options_.amplify = amplify;
      impl_->recompute_kernel();
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
    
    const ChannelsDesc::Transform &UnaryKernel::get_transform () const {
      return impl_->transform_;
    }
    
    void UnaryKernel::process (const Texture &source, const Texture &destination) {
      Kernel::process(source, destination);
    }
}