//
// Created by denn nevera on 2019-10-23.
//

#include "dehancer/gpu/spaces/StreamSpaceCache.h"
#include "dehancer/Log.h"

namespace dehancer {

    void stream_space_cache::invalidate() {
        std::unique_lock<std::mutex> lock(mutex_);
        clut_cache_.clear();
    }

    size_t stream_space_cache::get_hash(void *command_queue,
                                        const StreamSpace& space,
                                        StreamSpace::Direction direction) const {
        return
                std::hash<std::string>()(space.id)
                + std::hash<size_t>()(static_cast<size_t>(direction)) * 1000
                + std::hash<size_t>()(static_cast<size_t>(space.type)) * 10000
                + reinterpret_cast<std::size_t>(command_queue);

    }

    std::shared_ptr<CLut> stream_space_cache::get_lut(
            void *command_queue,
            const StreamSpace& space,
            StreamSpace::Direction direction) {

        const ocio::LutParameters* transform_lut = nullptr;

        switch (direction) {
            case StreamSpace::Direction::forward:
                if (!space.transform_lut.forward.enabled)
                    return nullptr;
                transform_lut = &space.transform_lut.forward;
                break;
            case StreamSpace::Direction::inverse:
                if (!space.transform_lut.inverse.enabled)
                    return nullptr;
                transform_lut = &space.transform_lut.inverse;
                break;
            case StreamSpace::Direction::none:
                return nullptr;
        }

        if (!transform_lut)
            return nullptr;

        std::unique_lock<std::mutex> lock(mutex_);

        size_t hash = get_hash(command_queue,space,direction);

        const auto it = clut_cache_.find(hash);

        if (it == clut_cache_.end())
        {

            auto clut = std::make_shared<CLutCubeInput>(command_queue);

            clut->
                    load_from_data(
                    transform_lut->data,
                    transform_lut->size,
                    transform_lut->channels);

            clut_cache_[hash] = clut;

            return clut;
        }
        else {

            return  it->second;
        }
    }
}
