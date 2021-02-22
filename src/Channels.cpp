//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/Channels.h"
#include "dehancer/gpu/Log.h"

#include <cmath>

namespace dehancer {
    
    static dehancer::ChannelsDesc::Transform options_one = {
            .slope = {8.0f,4,0,0},
            .offset = {128.0f,128,0,0},
            .enabled = {true,true,false,false},
            .direction = dehancer::ChannelsDesc::TransformDirection::forward,
            .mask = nullptr
    };
    
    ChannelsDesc::Scale2D ChannelsDesc::default_scale = {(Scale){1.0f, 1.0f}, {1.0f,1.0f}, {1.0f,1.0f}, {1.0f,1.0f}};
    
    Channels ChannelsDesc::make (const void *command_queue) const {
      return ChannelsHolder::Make(command_queue, *this);
    }
    
    namespace impl {
        
        struct ChannelsHolder: public dehancer::ChannelsHolder, public dehancer::Command {
            
            std::shared_ptr<std::array<Memory,4>> channels_;
            ChannelsDesc desc_;
            std::shared_ptr<std::array<ChannelsDesc,4>> channel_descs_;
            
            size_t get_width(int index) const override { return channel_descs_->at(index).width; };
            size_t get_height(int index) const override {return channel_descs_->at(index).height;};
    
            ChannelsDesc::Scale get_scale(int index) const override { return desc_.scale.at(index);}
            
            ChannelsDesc get_desc() const override { return desc_;}
            
            Memory& at(int index) override { return channels_->at(index);};
            const Memory& at(int index) const override { return channels_->at(index);};
            [[nodiscard]] size_t size() const override { return channels_->size(); };
            
            ChannelsHolder(const void *command_queue, const ChannelsDesc& desc):
                    Command(command_queue),
                    channels_(std::make_shared<std::array<Memory,4>>()),
                    desc_(desc),
                    channel_descs_({
                                           std::make_shared<std::array<ChannelsDesc,4>>()
                                   })
            {
              
              int i = 0;
              for(auto& c: *channel_descs_){
                c = desc_;
                c.width = std::floor((float )c.width * c.scale.at(i).x);
                c.height = std::floor((float )c.height * c.scale.at(i).y);
                ++i;
              }
              
              i = 0;
              for (auto& c : *channels_) {
                auto size = sizeof(float)*channel_descs_->at(i).width*channel_descs_->at(i).height;
                if (size==0) continue;
                c = MemoryHolder::Make(get_command_queue(),size);
                ++i;
              }
            }
    
            ~ChannelsHolder() override {
              #ifdef PRINT_DEBUG
              dehancer::log::print(" ### ~ChannelsHolder(base): %p: %ix%i", this, desc_.width, desc_.height);
              #endif
            }
        };
    }
    
    Channels ChannelsHolder::Make(const void *command_queue,
                                  size_t width,
                                  size_t height) {
      ChannelsDesc desc = {
              .width = width,
              .height = height
      };
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc);
    }
    
    Channels ChannelsHolder::Make (const void *command_queue, const ChannelsDesc &desc) {
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc);
    }
    
    
    ChannelsInput::ChannelsInput(const void *command_queue,
                                 const Texture &texture,
                                 const ChannelsDesc::Transform& transform,
                                 ChannelsDesc::Scale2D scale,
                                 bool wait_until_completed,
                                 const std::string& library_path):
            Kernel(command_queue,
                   "image_to_one_channel",
                   texture,
                   nullptr,
                   wait_until_completed,
                   library_path),
            desc_({
                          .width = texture ? texture->get_width() : 0,
                          .height = texture ? texture->get_height() : 0,
                          .scale = scale
                  }),
            channels_(nullptr),
            has_mask_(transform_.mask != nullptr),
            transform_(transform)
    {
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
    }
    
    void ChannelsInput::process () {
      
      if (!channels_) {
        channels_ = desc_.make(get_command_queue());
      }
      
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      
      for (int j = 0; j < channels->size(); ++j) {
        
        execute([this, channels, j](CommandEncoder& encoder){
            
            encoder.set(get_source(),0);
            
            encoder.set(channels->at(j),1);
            
            int cw = channels->get_width(j);
            int ch = channels->get_height(j);
            
            encoder.set(cw, 2);
            encoder.set(ch, 3);
            
            encoder.set(j, 4);
            
            encoder.set(transform_.slope[j],5);
            encoder.set(transform_.offset[j],6);
            
            if (transform_.flags.in_enabled)
              encoder.set(transform_.enabled[j],7);
            else
              encoder.set(false,7);
    
            encoder.set(transform_.direction ,8);
            encoder.set(transform_.type ,9);
            
            encoder.set(has_mask_ , 10);
            encoder.set(mask_ , 11);
            
            CommandEncoder::Size size = {
                    .width = channels->get_width(j),
                    .height = channels->get_height(j),
                    .depth = 1
            };
            
            return size;
        });
      }
    }
    
    void ChannelsInput::process (const Texture &source, const Texture &destination) {
      Kernel::process(source, destination);
    }
    
    
    void ChannelsInput::set_source (const Texture &source) {
      Kernel::set_source(source);
      if (!source) {
        channels_ = nullptr;
        return;
      }
      
      desc_.width = source->get_width();
      desc_.height = source->get_height();
      
      if (channels_
          &&
          (source->get_width()!=channels_->get_desc().width
           ||
           source->get_height()!=channels_->get_desc().height))
        channels_ = nullptr;
    }
    
    void ChannelsInput::set_destination (const Texture &destination) {
      Kernel::set_destination(nullptr);
    }
    
    void ChannelsInput::set_transform (const ChannelsDesc::Transform &transform) {
      transform_ = transform;
      has_mask_ = transform_.mask != nullptr;
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
      else
        mask_ = transform_.mask;
    }
    
    const ChannelsDesc::Transform &ChannelsInput::get_transform () const {
      return transform_;
    }
    
    void ChannelsInput::set_scale (ChannelsDesc::Scale2D scale) {
      bool do_recreate_channels = false;
      for (int i = 0; i < scale.size(); ++i) {
        if (desc_.scale.at(i).x!=scale.at(i).y || desc_.scale.at(i).y!=scale.at(i).y) {
          do_recreate_channels = true;
          break;
        }
      }
      desc_.scale = scale;
      if (do_recreate_channels){
        channels_ = nullptr;
      }
    }
    
    
    ChannelsOutput::ChannelsOutput(const void *command_queue,
                                   const Texture& destination,
                                   const Channels& channels,
                                   const ChannelsDesc::Transform& transform,
                                   bool wait_until_completed,
                                   const std::string& library_path):
            Kernel(command_queue,
                   "one_channel_to_image",
                   nullptr,
                   destination,
                   wait_until_completed,
                   library_path),
            channels_(channels),
            has_mask_(transform_.mask != nullptr),
            transform_(transform)
    {
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
    }
    
    void ChannelsOutput::process () {
     
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      
      
      for (int j = 0; j < channels->size(); ++j) {
        
        auto channel = channels->at(j);
        
        execute([this, channels, &channel, j](CommandEncoder& encoder){
            
            encoder.set(get_destination(),0);
            
            encoder.set(get_destination(),1);
            
            encoder.set(channel,2);
            
            int cw = channels->get_width(j);
            int ch = channels->get_height(j);
            
            encoder.set(cw, 3);
            encoder.set(ch, 4);
            
            encoder.set(j, 5);
            
            encoder.set(transform_.slope[j],6);
            encoder.set(transform_.offset[j],7);
            
            if (transform_.flags.out_enabled)
              encoder.set(transform_.enabled[j],8);
            else
              encoder.set(false,8);
            
            encoder.set(transform_.direction ,9);
    
            encoder.set(transform_.type ,10);
    
            encoder.set(has_mask_ , 11);
            encoder.set(mask_ , 12);
            
            return CommandEncoder::Size::From(get_destination());
        });
      }
    }
    
    void ChannelsOutput::process (const Texture &source, const Texture &destination) {
      Kernel::process(source, destination);
    }
    
    void ChannelsOutput::set_destination (const Texture &destination) {
      Kernel::set_destination(destination);
    }
    
    void ChannelsOutput::set_source (const Texture &source) {
      Kernel::set_source(nullptr);
    }
    
    void ChannelsOutput::set_transform (const ChannelsDesc::Transform &transform) {
      transform_ = transform;
      has_mask_ = transform_.mask != nullptr;
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
      else {
        mask_ = transform_.mask;
      }
    }
    
    const ChannelsDesc::Transform &ChannelsOutput::get_transform () const {
      return transform_;
    }
    
    void ChannelsOutput::set_channels (const Channels &channels) {
      channels_ = channels;
    }
  
}