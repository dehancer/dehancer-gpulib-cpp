#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"

namespace dehancer::device {
std::string get_lib_path() {
	return CUDA_KERNELS_LIBRARY;
}

std::size_t get_lib_source(std::string& source) {
	source.clear();
	return 0;
}
}  // namespace dehancer::device

namespace {
struct Queue {
	void* handle = dehancer::DeviceCache::Instance().get_default_command_queue();

	~Queue() {
		if (handle)
			dehancer::DeviceCache::Instance().return_command_queue(handle);
	}
};

struct Visits {
	unsigned int* data = nullptr;

	~Visits() {
		if (data)
			cudaFree(data);
	}
};

struct CurrentContext {
	~CurrentContext() {
		CUcontext previous = nullptr;
		cuCtxPopCurrent(&previous);
	}
};
}  // namespace

TEST(CUDA, KernelBlockLimitAndCoverage) {
	Queue queue;
	ASSERT_NE(queue.handle, nullptr);
	CUcontext native_context = nullptr;
	ASSERT_EQ(cuStreamGetCtx(static_cast<CUstream>(queue.handle), &native_context), CUDA_SUCCESS);
	ASSERT_EQ(cuCtxPushCurrent(native_context), CUDA_SUCCESS);
	CurrentContext current;
	// Both a fresh lookup and a cached function must honor launch bounds.
	dehancer::Function first(queue.handle, "kernel_limited_block");
	dehancer::Function cached(queue.handle, "kernel_limited_block");
	EXPECT_EQ(first.get_block_max_size(), 128u);
	EXPECT_EQ(cached.get_block_max_size(), 128u);

	for (const auto& extent : {std::array<int, 3>{1, 1, 1}, {71, 1, 1}, {37, 29, 1}, {129, 65, 3}}) {
		const auto width = extent[0], height = extent[1], depth = extent[2];
		const auto count = static_cast<size_t>(width) * height * depth;
		const auto size = first.ask_compute_size(width, height, depth);
		CUcontext current_context = nullptr;
		ASSERT_EQ(cuCtxGetCurrent(&current_context), CUDA_SUCCESS);
		EXPECT_EQ(current_context, native_context);
		ASSERT_GT(size.block.width, 0u);
		ASSERT_GT(size.block.height, 0u);
		EXPECT_LE(size.block.width * size.block.height * size.block.depth, 128u);
		EXPECT_GE(size.grid.width * size.block.width, static_cast<size_t>(width));
		EXPECT_GE(size.grid.height * size.block.height, static_cast<size_t>(height));
		EXPECT_EQ(size.grid.depth, static_cast<size_t>(depth));

		Visits visits;
		ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&visits.data), count * sizeof(unsigned int)), cudaSuccess);
		ASSERT_EQ(cudaMemset(visits.data, 0, count * sizeof(unsigned int)), cudaSuccess);
		cached.execute([&](dehancer::CommandEncoder& encoder) {
			EXPECT_EQ(encoder.get_block_max_size(), 128u);
			encoder.set(&visits.data, sizeof(visits.data), 0);
			encoder.set(width, 1);
			encoder.set(height, 2);
			encoder.set(depth, 3);
			return dehancer::CommandEncoder::Size::From(width, height, depth);
		});
		std::vector<unsigned int> result(count);
		ASSERT_EQ(cudaMemcpy(result.data(), visits.data, count * sizeof(unsigned int), cudaMemcpyDeviceToHost), cudaSuccess);
		for (const auto visits_per_pixel : result)
			ASSERT_EQ(visits_per_pixel, 1u);
	}
}
