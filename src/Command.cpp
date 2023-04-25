//
// Created by denn nevera on 09/11/2020.
//

#include "dehancer/gpu/Command.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "src/platforms/metal/Command.h"
#elif defined(DEHANCER_GPU_CUDA)
#include "src/platforms/cuda/Command.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "src/platforms/opencl/Command.h"
#endif


#ifdef DEHANCER_GPU_PLATFORM

namespace dehancer {

    bool Command::WAIT_UNTIL_COMPLETED = false;
    TextureDesc::PixelFormat Command::pixel_format_1d = TextureDesc::PixelFormat::rgba32float;
    TextureDesc::PixelFormat Command::pixel_format_2d = TextureDesc::PixelFormat::rgba32float;
    
    #if defined(IOS_SYSTEM)
    
    TextureDesc::PixelFormat Command::pixel_format_3d = TextureDesc::PixelFormat::rgba16float;
    
    #elif defined(DEHANCER_3DLUT_32FLOAT) || defined(DEHANCER_GPU_CUDA) // TODO: Cuda trilinear interpolation is not supported yet
    
    TextureDesc::PixelFormat Command::pixel_format_3d = TextureDesc::PixelFormat::rgba32float;
    
    #elif defined(DEHANCER_GPU_OPENCL)
    
    TextureDesc::PixelFormat Command::pixel_format_3d = TextureDesc::PixelFormat::rgba32float;
    
    #else
    
    TextureDesc::PixelFormat Command::pixel_format_3d = TextureDesc::PixelFormat::rgba16float;
    
    #endif
    
    namespace impl {
        class Command: public dehancer::DEHANCER_GPU_PLATFORM::Command {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Command::Command;
        };
    }

    Command::Command(const void *command_queue, bool wait_until_completed):
    impl_(std::make_shared<impl::Command>(command_queue,wait_until_completed))
    {}

    Texture Command::make_texture(size_t width, size_t height, size_t depth) {
      return impl_->make_texture(width,height,depth);
    }

    void Command::enable_wait_completed(bool enable) {
      impl_->enable_wait_completed(enable);
    }

    bool Command::get_wait_completed() {
      return impl_->get_wait_completed();
    }

    const void *Command::get_command_queue() const {
#if DEHANCER_GPU_OPENCL
      return impl_->get_cl_command_queue();
#else
      return impl_->get_command_queue();
#endif
    }

    void *Command::get_command_queue() {
#if DEHANCER_GPU_OPENCL
      return impl_->get_cl_command_queue();
#else
      return impl_->get_command_queue();
#endif
    }
    
    void Command::set_wait_completed (bool enable) {
    
    }
    
    TextureInfo Command::get_texture_info (TextureDesc::Type texture_type) const {
      return impl_->get_texture_info(texture_type);
    }
    
    Command::~Command() = default;
}

#endif