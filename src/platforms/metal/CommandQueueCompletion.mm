#include <dehancer/gpu/metal/CommandQueueCompletion.h>
#import <Metal/Metal.h>

#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace dehancer::metal {
namespace {
std::shared_ptr<void> retain_object(id object) {
    [object retain];
    return {(__bridge void*)object, [](void* value) { [(__bridge id)value release]; }};
}

struct PendingCommand {
    std::shared_ptr<void> buffer;
    std::string kernel;

    id<MTLCommandBuffer> native() const { return (__bridge id<MTLCommandBuffer>)buffer.get(); }
    bool completed() const { return native().status >= MTLCommandBufferStatusCompleted; }
    void record_error(std::string& failure) const {
        if (!failure.empty() || native().status == MTLCommandBufferStatusCompleted) return;
        const char* description = [[native().error localizedDescription] UTF8String];
        failure = "Metal kernel " + kernel + " failed: " +
                  (description ? description : "unknown command error");
    }
};
} // namespace

struct CommandQueueCompletion::Impl {
    struct Registry {
        std::mutex mutex;
        std::unordered_map<const void*, std::weak_ptr<Impl>> queues;
    };
    static std::shared_ptr<Impl> lookup(const void* queue, bool create) {
        static Registry registry;
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.queues.find(queue);
        if (found != registry.queues.end()) {
            if (auto state = found->second.lock()) return state;
            registry.queues.erase(found);
        }
        if (!create) return {};
        for (auto it = registry.queues.begin(); it != registry.queues.end();) {
            if (it->second.expired()) it = registry.queues.erase(it);
            else ++it;
        }
        auto state = std::make_shared<Impl>(queue);
        registry.queues.emplace(queue, state);
        return state;
    }

    explicit Impl(const void* queue) : queue_owner(retain_object((__bridge id)queue)) {}

    std::shared_ptr<void> queue_owner;
    std::mutex mutex, wait_mutex;
    std::deque<PendingCommand> pending;
    std::string failure;

    void track(id<MTLCommandBuffer> buffer, const std::string& kernel) {
        std::lock_guard<std::mutex> lock(mutex);
        // Retire completed buffers on submission so long-running asynchronous
        // clients retain only in-flight work, plus the first error message.
        while (!pending.empty() && pending.front().completed()) {
            pending.front().record_error(failure);
            pending.pop_front();
        }
        pending.push_back({retain_object(buffer), kernel});
    }

    void wait() {
        std::lock_guard<std::mutex> drain(wait_mutex);
        std::deque<PendingCommand> commands;
        std::string first_failure;
        {
            std::lock_guard<std::mutex> lock(mutex);
            commands.swap(pending);
            first_failure.swap(failure);
        }
        // Drain the whole snapshot before reporting an error, keeping every
        // command's resources alive even when an earlier command has failed.
        for (const auto& command : commands) {
            [command.native() waitUntilCompleted];
            command.record_error(first_failure);
        }
        if (!first_failure.empty()) throw std::runtime_error(first_failure);
    }
};

CommandQueueCompletion::CommandQueueCompletion(const void* queue) {
    if (!queue) throw std::invalid_argument("Metal command queue is null");
    impl_ = Impl::lookup(queue, true);
}
CommandQueueCompletion::~CommandQueueCompletion() = default;
void CommandQueueCompletion::wait() { impl_->wait(); }
void CommandQueueCompletion::track(const void* queue, void* buffer, const std::string& kernel) {
    if (auto state = Impl::lookup(queue, false))
        state->track((__bridge id<MTLCommandBuffer>)buffer, kernel);
}

} // namespace dehancer::metal
