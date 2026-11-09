#pragma once

#include <memory>
#include <string>

namespace dehancer::metal {

// Tracks GPULib kernel submissions while at least one handle for the queue lives.
// Handles on the same queue share pending commands and the first unreported error.
class CommandQueueCompletion final {
public:
    explicit CommandQueueCompletion(const void* command_queue);
    ~CommandQueueCompletion();

    // Waits for all GPULib commands submitted before this call, then throws on
    // failure. Errors from completed commands survive later successful commands.
    void wait();

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class Function;
    static void track(const void* queue, void* command_buffer, const std::string& kernel);
};

} // namespace dehancer::metal
