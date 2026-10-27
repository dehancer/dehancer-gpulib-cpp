#pragma once

#include "Context.h"
#include "dehancer/gpu/Command.h"

namespace dehancer::cuda {

class Command : public cuda::Context, public std::enable_shared_from_this<Command> {
public:
	explicit Command(const void* command_queue, bool wait_until_completed = dehancer::Command::WAIT_UNTIL_COMPLETED);
	Texture make_texture(size_t width, size_t height, size_t depth);

	void enable_wait_completed(bool enable) {
		wait_until_completed_ = enable;
	};

	[[nodiscard]] bool get_wait_completed() const {
		return wait_until_completed_;
	}

	void set_wait_completed(bool value) {
		wait_until_completed_ = value;
	}

private:
	bool wait_until_completed_;
};
}  // namespace dehancer::cuda
