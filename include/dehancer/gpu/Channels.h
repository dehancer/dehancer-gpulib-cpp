//
// Created by denn nevera on 30/11/2020.
//

#pragma once

#include <array>
#include "dehancer/gpu/Memory.h"
#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/TextureIO.h"

namespace dehancer {

    struct ChannelsHolder;

    /***
     * Memory pointer object
     */
    using Channels = std::shared_ptr<ChannelsHolder>;

    namespace impl {
        class ChannelsHolder;
    }

    struct ChannelsHolder: public std::enable_shared_from_this<ChannelsHolder> {

    public:

        static Channels Make(const void *command_queue, size_t width, size_t height);

        virtual size_t get_width() const = 0;
        virtual size_t get_height() const = 0;

        virtual Memory& at(int index) = 0;
        virtual const Memory& at(int index) const = 0;
        [[nodiscard]] virtual inline size_t size() const = 0;

        Channels get_ptr() { return shared_from_this(); }

        virtual ~ChannelsHolder() = default;

    protected:
        ChannelsHolder() = default;
    };

    class ChannelsInput: public Kernel {

    public:

        explicit ChannelsInput(const void *command_queue,
                               const Texture& source,
                               bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                               const std::string& library_path = ""
        );

        [[nodiscard]] const Channels& get_channels() const { return channels_;}
        void setup(CommandEncoder &encode) override;

    private:
        Channels channels_;
    };

    class ChannelsOutput: public Kernel {

    public:

        explicit ChannelsOutput(const void *command_queue,
                                const Texture& destination,
                                const Channels& channels,
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string& library_path = ""
        );

        void setup(CommandEncoder &encode) override;

    private:
        Channels channels_;
    };
}

