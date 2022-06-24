//
// Created by denn nevera on 10/11/2020.
//

#include <dehancer/gpu/Function.h>

#include "platforms/PlatformConfig.h"
#include "platforms/PlatformConfig.h"

#if defined(DEHANCER_GPU_METAL)
#include "platforms/metal/Function.h"
#include "src/platforms/metal/Command.h"
#elif defined(DEHANCER_GPU_CUDA)
#include "platforms/cuda/Function.h"
#include "src/platforms/cuda/Command.h"
#elif defined(DEHANCER_GPU_OPENCL)
#include "platforms/opencl/Function.h"
#include "src/platforms/opencl/Command.h"
#endif


#ifdef DEHANCER_GPU_PLATFORM

namespace dehancer {

    namespace impl {

        class Function: public dehancer::DEHANCER_GPU_PLATFORM::Function {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Function::Function;
        };

        class Command: public dehancer::DEHANCER_GPU_PLATFORM::Command {
        public:
            using dehancer::DEHANCER_GPU_PLATFORM::Command::Command;
        };
    }

    Function::Function(const void *command_queue,
                       const std::string &kernel_name,
                       bool wait_until_completed,
                       const std::string &library_path) :
    Command(command_queue,wait_until_completed),
    impl_(std::make_shared<impl::Function>(Command::impl_.get(), kernel_name, library_path))
    {
    }

    void Function::execute(const Function::EncodeHandler &block) {
      impl_->execute(block);
    }
    
    void Function::execute (CommandEncoder::ComputeSize compute_size,
                            const Function::VoidEncodeHandler &block) {
      impl_->execute(compute_size, block);
    }
    
    const std::vector<dehancer::Function::ArgInfo> & Function::get_arg_list() const {
      return impl_->get_arg_info_list();
    }

    const std::string& Function::get_name() const {
      return impl_->get_name();
    }
    
    const std::string &Function::get_library_path () const {
      return impl_->get_library_path();
    }
    
    CommandEncoder::ComputeSize Function::ask_compute_size (CommandEncoder::Size texture_size) {
      return ask_compute_size(texture_size.width, texture_size.height, texture_size.depth);
    }
    
    CommandEncoder::ComputeSize Function::ask_compute_size (const Texture &source) {
      return ask_compute_size(source->get_desc().width, source->get_desc().height, source->get_desc().depth);
    }
    
    CommandEncoder::ComputeSize Function::ask_compute_size (size_t width, size_t height, size_t depth) const {
      return impl_->ask_compute_size(width,height,depth);
    }
    
    size_t Function::get_block_max_size () const {
      return impl_->get_block_max_size();
    }
}

#endif
