//
// Created by denn nevera on 2019-08-02.
//

#include "dehancer/gpu/operations/ResizeKernel.h"

namespace dehancer {

    static std::vector<std::string> _resize_kernel_name_ = {
      "kernel_gauss",
      "kernel_lanczos",
    };

    ResizeKernel::ResizeKernel(const void *command_queue,
                               const Texture &source,
                               const Texture &destination,
                               Mode mode,
                               float radius,
                               bool wait_until_completed,
                               const std::string &library_path) :
      Kernel(command_queue,
             _resize_kernel_name_[mode],
             source, destination, wait_until_completed, library_path),
      radius_(radius),
      mode_(mode)
    {}

    ResizeKernel::ResizeKernel (const void *command_queue,
                                Mode mode,
                                float radius,
                                bool wait_until_completed,
                                const std::string &library_path):
      ResizeKernel(command_queue, nullptr, nullptr, mode, radius, wait_until_completed, library_path)
    {}

    void ResizeKernel::process () {
      auto src = get_source();

      auto desc = src->get_desc();
      desc.width = get_destination()->get_desc().width;

      {
        auto tmp_ = desc.make(get_command_queue());
        execute([this, &src, &tmp_](CommandEncoder &encoder) {

            float2 direction = {radius_, 0.0f};

            encoder.set(src, 0);
            encoder.set(tmp_, 1);
            encoder.set(direction, 2);
            return CommandEncoder::Size::From(tmp_);
        });

        execute([this, &tmp_](CommandEncoder &encoder) {

            float2 direction = {0.0f, radius_};

            encoder.set(tmp_, 0);
            encoder.set(get_destination(), 1);
            encoder.set(direction, 2);
            return CommandEncoder::Size::From(get_destination());
        });
      }
    }

    void ResizeKernel::process (const Texture &source, const Texture &destination) {
      Kernel::process(source, destination);
    }

}