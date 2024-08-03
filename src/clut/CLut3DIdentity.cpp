//
// Created by denn nevera on 2019-07-23.
//

#include "dehancer/gpu/clut/CLut3DIdentity.h"

namespace dehancer {

    namespace impl {

        struct CLut3DIdentityImpl {
            size_t   lut_size;
            Texture  texture;
        };

    }

    CLut3DIdentity::CLut3DIdentity(
      const void *command_queue, size_t lut_size, bool wait_until_completed,
      const std::string &library_path):
      CLut3DIdentity(command_queue,
                     CLut::Options({
                                     .size = lut_size,
                                     .pixel_format = dehancer::Command::pixel_format_3d
                                   }),
                     wait_until_completed, library_path)
    {}

    CLut3DIdentity::CLut3DIdentity(
      const void *command_queue,
      CLut::Options options,
      bool wait_until_completed,
      const std::string &library_path):
      Function(command_queue, "kernel_make3DLut", wait_until_completed, library_path),
      CLut(),
      impl_(new impl::CLut3DIdentityImpl({
                                           .lut_size = options.size,
                                           .texture = nullptr
                                         }))
    {

      dehancer::TextureDesc desc = {
        .width = impl_->lut_size,
        .height = impl_->lut_size,
        .depth = impl_->lut_size,
        .pixel_format = options.pixel_format,
        .type = TextureDesc::Type::i3d,
        .mem_flags = TextureDesc::MemFlags::read_write
      };

      impl_->texture = desc.make(get_command_queue());

      execute([this](CommandEncoder& compute_encoder) {
          compute_encoder.set(impl_->texture,0);
          compute_encoder.set(float2({1,0}),1);
          return CommandEncoder::Size::From(impl_->texture);
      });

    }

    const Texture& CLut3DIdentity::get_texture() { return impl_->texture; };

    const Texture& CLut3DIdentity::get_texture() const  { return impl_->texture; };

    size_t CLut3DIdentity::get_lut_size() const { return impl_->lut_size; };

    CLut3DIdentity::~CLut3DIdentity() = default;
}