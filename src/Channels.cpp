//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/Channels.h"

namespace dehancer {

    namespace impl {
        struct ChannelsHolder: public dehancer::ChannelsHolder, public dehancer::Command {

            typedef std::shared_ptr<std::array<Memory,4>> Array;

            size_t get_width() const override { return width_; };
            size_t get_height() const override {return height_;};

            Memory& at(int index) override { return channels_->at(index);};
            const Memory& at(int index) const override { return channels_->at(index);};
            [[nodiscard]] size_t size() const override { return channels_->size(); };

            ChannelsHolder(const void *command_queue,
                           size_t width, size_t height):
                    Command(command_queue),
                    width_(width),
                    height_(height),
                    channels_(std::make_shared<std::array<Memory,4>>())
            {
              for (auto & c : *channels_) {
                c = MemoryHolder::Make(get_command_queue(),sizeof(float)*width_*height_);
              }
            }

            size_t width_;
            size_t height_;
            std::shared_ptr<std::array<Memory,4>> channels_;
        };
    }

    Channels ChannelsHolder::Make(const void *command_queue,
                                  size_t width,
                                  size_t height) {
      return std::make_shared<impl::ChannelsHolder>(command_queue,width,height);
    }


    ChannelsInput::ChannelsInput(const void *command_queue,
                                 const Texture &texture,
                                 bool wait_until_completed,
                                 const std::string& library_path):
            Kernel(command_queue,
                   "image_to_channels",
                   texture,
                   nullptr,
                   wait_until_completed,
                   library_path),
            channels_(ChannelsHolder::Make(command_queue,texture->get_width(),texture->get_height()))
    {}

    void ChannelsInput::setup(CommandEncoder &command)  {
      for (int i = 0; i <channels_->size(); ++i) {
        command.set(channels_->at(i),i+1);
      }
    }

    ChannelsOutput::ChannelsOutput(const void *command_queue,
                                   const Texture& destination,
                                   const Channels& channels,
                                   bool wait_until_completed,
                                   const std::string& library_path):
            Kernel(command_queue,
                   "channels_to_image",
                   nullptr,
                   destination,
                   wait_until_completed,
                   library_path),
            channels_(channels)
    {}

    void ChannelsOutput::setup(CommandEncoder &command) {
      for (int i = 0; i <channels_->size(); ++i) {
        command.set(channels_->at(i),i+1);
      }
    }
}