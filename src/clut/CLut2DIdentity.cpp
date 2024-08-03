//
// Created by denn nevera on 2019-07-23.
//

#include "dehancer/gpu/clut/CLut2DIdentity.h"

namespace dehancer {


    namespace impl {
        struct CLut2DIdentityImpl {
            Texture  texture;
            size_t   level;
            size_t   lut_size;
        };
    }

    CLut2DIdentity::CLut2DIdentity(
      const void *command_queue,
      size_t lut_size,
      bool wait_until_completed,
      const std::string &library_path):
      CLut2DIdentity(command_queue,
                     CLut::Options({
                       .size = lut_size,
                       .pixel_format = dehancer::Command::pixel_format_2d
                     }),
                     wait_until_completed, library_path)
    {}

    CLut2DIdentity::CLut2DIdentity(
      const void *command_queue,
      CLut::Options options,
      bool wait_until_completed ,
      const std::string &library_path ):
      Function(command_queue, "kernel_make2DLut", wait_until_completed, library_path),
      CLut(),
      impl_(new impl::CLut2DIdentityImpl({
                                           .texture =nullptr,
                                           .level = (size_t)sqrtf((float)options.size),
                                           .lut_size = options.size
                                         }))
    {

      auto level_ = impl_->level;
      auto dimension = level_*level_*level_;

      dehancer::TextureDesc const desc = {
        .width = dimension,
        .height = dimension,
        .depth = 1,
        .pixel_format = options.pixel_format,
        .type = TextureDesc::Type::i2d,
        .mem_flags = TextureDesc::MemFlags::read_write
      };

      impl_->texture = desc.make(get_command_queue());

      execute([this](CommandEncoder& compute_encoder) {

          compute_encoder.set(impl_->texture, 0);
          compute_encoder.set(float2({1,0}),1);
          compute_encoder.set(uint(impl_->level),2);

          return CommandEncoder::Size::From(impl_->texture);
      });

    }

    const Texture& CLut2DIdentity::get_texture() {
      return impl_->texture;
    };

    const Texture& CLut2DIdentity::get_texture() const {
      return impl_->texture;
    };

    size_t CLut2DIdentity::get_lut_size() const{
      return impl_->lut_size;
    };

    CLut2DIdentity::~CLut2DIdentity() = default;
}