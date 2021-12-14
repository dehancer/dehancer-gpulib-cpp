//
// Created by denn nevera on 2019-10-23.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/StreamSpace.h"
#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/clut/CLutCubeInput.h"

#include <mutex>

namespace dehancer {

    class stream_space_cache {

    public:
        typedef std::unordered_map<std::size_t, std::shared_ptr<CLutCubeInput>> LutCache;

        stream_space_cache():clut_cache_(){};

        void invalidate();
        [[nodiscard]] std::shared_ptr<CLut> get_lut(void *command_queue, const StreamSpace& space, StreamSpaceDirection direction);

        void lock() { user_mutex_.lock(); }
        void unlock() { user_mutex_.unlock(); }

    private:
        StreamSpace           space_;
        LutCache              clut_cache_;

        std::mutex mutex_;
        std::mutex user_mutex_;

        static size_t get_hash(void *command_queue,
                        const StreamSpace& space,
                        StreamSpaceDirection direction) ;
    };

    class StreamSpaceCache: public ControlledSingleton<stream_space_cache>{
    public:
        StreamSpaceCache() = default;
    };

}