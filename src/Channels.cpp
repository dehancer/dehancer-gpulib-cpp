//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/Channels.h"

namespace dehancer {
    
    Channels ChannelDesc::make (const void *command_queue) const {
      return ChannelsHolder::Make(command_queue, *this);
    }
    
    namespace impl {
        struct ChannelsHolder: public dehancer::ChannelsHolder, public dehancer::Command {
            
            
            typedef std::shared_ptr<std::array<Memory,4>> Array;
            
            size_t get_width() const override { return desc_.width; };
            size_t get_height() const override {return desc_.height;};
            
            void set_transform(const ChannelDesc::Transform &transform) override {
              desc_.transform = {
                      .slope = transform.slope,
                      .offset = transform.offset,
                      .enabled = transform.enabled,
                      .direction = transform.direction
              };
            }
            
            const ChannelDesc::Transform & get_transform() const override {
              return desc_.transform;
            }
            
            Memory& at(int index) override { return channels_->at(index);};
            const Memory& at(int index) const override { return channels_->at(index);};
            [[nodiscard]] size_t size() const override { return channels_->size(); };

//            ChannelsHolder(const void *command_queue,
//                           size_t width, size_t height):
//                    Command(command_queue),
//                    channels_(std::make_shared<std::array<Memory,4>>()),
//                    desc_({
//                                  .width = width,
//                                  .height = height,
//                                  .transform = {
//                                          .enabled = {false,false,false,false}
//                                  }
//                          })
//            {
//              auto size = sizeof(float)*desc_.width*desc_.height;
//              if (size==0) return;
//              for (auto & c : *channels_) {
//                c = MemoryHolder::Make(get_command_queue(),size);
//              }
//            }
            
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
      //return std::make_shared<impl::ChannelsHolder>(command_queue,width,height);
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
                    //texture ? texture->get_width() : 0,
                    //texture ? texture->get_height() : 0)
                                           (ChannelDesc) {
                                                   .width = texture ? texture->get_width() : 0,
                                                   .height = texture ? texture->get_height() : 0,
                                                   .transform = transform
                                           }
            ))
    {
      //channels_->set_transform(transform);
    }
    
    void ChannelsInput::setup(CommandEncoder &command)  {
      int i = 0;
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      for (; i <channels->size(); ++i) {
        command.set(channels->at(i),i+1);
      }
      command.set(channels->desc_.transform.slope,i+1);
      command.set(channels->desc_.transform.offset,i+2);
      command.set(channels->desc_.transform.enabled,i+3);
      command.set(channels->desc_.transform.direction ,i+4);
    }
    
    void ChannelsInput::set_source (const Texture &source) {
      Kernel::set_source(source);
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      auto desc = channels->desc_;
      size_t width = source?source->get_width():0;
      size_t height = source?source->get_height():0;
      channels_ = ChannelsHolder::Make(get_command_queue(),
                                       width,
                                       height);
      channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      desc.width = width;
      desc.height = height;
      channels->desc_ = desc;
    }
    
    void ChannelsInput::set_destination (const Texture &destination) {
      Kernel::set_destination(nullptr);
    }
    
    void ChannelsInput::set_transform (const ChannelDesc::Transform &transform) {
      channels_->set_transform(transform);
    }
    
    const ChannelDesc::Transform &ChannelsInput::get_transform () const {
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      return channels->desc_.transform;
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
            channels_(channels)
    {
      auto *c = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      c->desc_.transform = transform;
    }
    
    void ChannelsOutput::setup(CommandEncoder &command) {
      int i = 0;
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      for (; i <channels->size(); ++i) {
        command.set(channels->at(i),i+1);
      }
      command.set(channels->desc_.transform.slope,i+1);
      command.set(channels->desc_.transform.offset,i+2);
      command.set(channels->desc_.transform.enabled,i+3);
      command.set(channels->desc_.transform.direction,i+4);
    }
    
    void ChannelsOutput::set_destination (const Texture &destination) {
      Kernel::set_destination(destination);
    }
    
    void ChannelsOutput::set_source (const Texture &source) {
      Kernel::set_source(nullptr);
    }
    
    void ChannelsOutput::set_transform (const ChannelDesc::Transform &transform) {
      channels_->set_transform(transform);
    }
    
    const ChannelDesc::Transform &ChannelsOutput::get_transform () const {
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      return channels->desc_.transform;
    }
  
}