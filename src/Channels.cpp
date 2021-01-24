//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/Channels.h"

namespace dehancer {
    
    static dehancer::ChannelDesc::Transform options_one = {
            .slope = {8.0f,4,0,0},
            .offset = {128.0f,128,0,0},
            .enabled = {true,true,false,false},
            .direction = dehancer::ChannelDesc::TransformDirection::forward,
            .mask = nullptr
    };
    
    
    Channels ChannelDesc::make (const void *command_queue) const {
      return ChannelsHolder::Make(command_queue, *this);
    }
    
    namespace impl {
        struct ChannelsHolder: public dehancer::ChannelsHolder, public dehancer::Command {
            
            typedef std::shared_ptr<std::array<Memory,4>> Array;
            
            size_t get_width() const override { return desc_.width; };
            size_t get_height() const override {return desc_.height;};
            
            Memory& at(int index) override { return channels_->at(index);};
            const Memory& at(int index) const override { return channels_->at(index);};
            [[nodiscard]] size_t size() const override { return channels_->size(); };
            
            ChannelsHolder(const void *command_queue, const ChannelDesc& desc):
                    Command(command_queue),
                    channels_(std::make_shared<std::array<Memory,4>>()),
                    desc_(desc)
            {
              auto size = sizeof(float)*desc_.width*desc_.height;
              if (size==0) return;
              for (auto & c : *channels_) {
                c = MemoryHolder::Make(get_command_queue(),size);
              }
            }
            std::shared_ptr<std::array<Memory,4>> channels_;
            ChannelDesc desc_;
        };
      
    }
    
    Channels ChannelsHolder::Make(const void *command_queue,
                                  size_t width,
                                  size_t height) {
      ChannelDesc desc = {
              .width = width,
              .height = height
      };
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc);
    }
    
    Channels ChannelsHolder::Make (const void *command_queue, const ChannelDesc &desc) {
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc);
    }
    
    
    ChannelsInput::ChannelsInput(const void *command_queue,
                                 const Texture &texture,
                                 const ChannelDesc::Transform& transform,
                                 bool wait_until_completed,
                                 const std::string& library_path):
            Kernel(command_queue,
                   "image_to_channels",
                   texture,
                   nullptr,
                   wait_until_completed,
                   library_path),
            channels_(ChannelsHolder::Make(command_queue,
                                           (ChannelDesc) {
                                                   .width = texture ? texture->get_width() : 0,
                                                   .height = texture ? texture->get_height() : 0
                                           }
            )),
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
    
    void ChannelsInput::setup(CommandEncoder &command)  {
      int i = 0;
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      for (; i <channels->size(); ++i) {
        command.set(channels->at(i),i+1);
      }
      command.set(transform_.slope,i+1);
      command.set(transform_.offset,i+2);
      command.set(transform_.enabled,i+3);
      command.set(transform_.direction ,i+4);
      command.set(has_mask_ , i + 5);
      command.set(mask_ ,i+6);
    }
    
    void ChannelsInput::set_source (const Texture &source) {
      Kernel::set_source(source);
      size_t width = source?source->get_width():0;
      size_t height = source?source->get_height():0;
      channels_ = ChannelsHolder::Make(get_command_queue(),
                                       width,
                                       height);
    }
    
    void ChannelsInput::set_destination (const Texture &destination) {
      Kernel::set_destination(nullptr);
    }
    
    void ChannelsInput::set_transform (const ChannelDesc::Transform &transform) {
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
    
    const ChannelDesc::Transform &ChannelsInput::get_transform () const {
      return transform_;
    }
    
    
    ChannelsOutput::ChannelsOutput(const void *command_queue,
                                   const Texture& destination,
                                   const Channels& channels,
                                   const ChannelDesc::Transform& transform,
                                   bool wait_until_completed,
                                   const std::string& library_path):
            Kernel(command_queue,
                   "channels_to_image",
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
    
    void ChannelsOutput::setup(CommandEncoder &command) {
      int i = 0;
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      for (; i <channels->size(); ++i) {
        command.set(channels->at(i),i+1);
      }
      command.set(transform_.slope,i+1);
      command.set(transform_.offset,i+2);
      command.set(transform_.enabled,i+3);
      command.set(transform_.direction,i+4);
      command.set(has_mask_, i + 5);
      command.set(mask_,i+6);
    }
    
    void ChannelsOutput::set_destination (const Texture &destination) {
      Kernel::set_destination(destination);
    }
    
    void ChannelsOutput::set_source (const Texture &source) {
      Kernel::set_source(nullptr);
    }
    
    void ChannelsOutput::set_transform (const ChannelDesc::Transform &transform) {
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
    
    const ChannelDesc::Transform &ChannelsOutput::get_transform () const {
      return transform_;
    }
    
    void ChannelsOutput::set_channels (const Channels &channels) {
      channels_ = channels;
    }
  
}