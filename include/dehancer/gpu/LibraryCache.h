//
// Created by dimich on 05.03.2024.
//

#ifndef DEHANCER_GPULIB_LIBRARYCACHE_H
#define DEHANCER_GPULIB_LIBRARYCACHE_H

#include <memory>

#include "dehancer/Common.h"
#include "dehancer/gpu/Command.h"
#include "dehancer/gpu/DeviceConfig.h"

#if defined(DEHANCER_GPU_OPENCL)

namespace dehancer {

namespace impl {
class GPULibraryCache;
}

class GPULibraryCache : public Command {
public:
	GPULibraryCache() = delete;

	explicit GPULibraryCache(const void* command_queue);

	virtual bool exists(const std::string& library_source);

	virtual bool compile_program(const std::string& library_source);

protected:
	std::shared_ptr<impl::GPULibraryCache> impl_;
};
}  // namespace dehancer
#endif

#endif  // DEHANCER_GPULIB_LIBRARYCACHE_H
