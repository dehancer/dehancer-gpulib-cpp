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
    
    struct ChannelDesc {
        
        enum TransformType:int {
            log_linear = 0
        };
    
        enum TransformDirection:int {
            forward = 0,
            inverse,
            none
        };
    
        struct Transform {
            TransformType            type = log_linear;
            dehancer::math::float4  slope = { 0.f,0.f,0.f,0.f };
            dehancer::math::float4 offset = { 1.0f,1.0f,1.0f,1.0f };
            dehancer::math::bool4 enabled = {false,false,false,false};
            TransformDirection  direction = none;
            Texture                  mask = nullptr;
            
            static Transform make(TransformType type, float slope, float offset, TransformDirection direction, bool enabled = true) {
              return {
                .type = type,
                .slope =  { slope,slope,slope,0.f },
                .offset = { offset,offset,offset,0.f },
                .enabled = {enabled,enabled,enabled,false},
                .direction = direction
              };
            };
        };
        
        size_t  width     = 0;
        size_t  height    = 0;
    
        Channels make(const void *command_queue) const;
      
    };
    
    struct ChannelsHolder: public std::enable_shared_from_this<ChannelsHolder> {
    
    public:
        
        static Channels Make(const void *command_queue, size_t width, size_t height);
        static Channels Make(const void *command_queue, const ChannelDesc& desc);
        
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
                               const ChannelDesc::Transform& transform = {},
                               bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                               const std::string& library_path = ""
        );
        
        [[nodiscard]] const Channels& get_channels() const { return channels_;}
        void setup(CommandEncoder &encode) override;
        
        void set_source(const Texture& source) override;
        void set_destination(const Texture& destination) override;
    
        virtual void set_transform(const ChannelDesc::Transform& transform);
        virtual const ChannelDesc::Transform& get_transform() const;
        
    private:
        Channels channels_;
        ChannelDesc::Transform transform_;
        bool has_mask_;
        Texture mask_;
    };
    
    class ChannelsOutput: public Kernel {
    
    public:
        
        using Kernel::Kernel;
        
        explicit ChannelsOutput(const void *command_queue,
                                const Texture& destination,
                                const Channels& channels,
                                const ChannelDesc::Transform& transform = {},
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string& library_path = ""
        );
        
        void setup(CommandEncoder &encode) override;
        void set_source(const Texture& source) override;
        void set_destination(const Texture& destination) override;
        virtual void set_transform(const ChannelDesc::Transform& transform);
        virtual void set_channels(const Channels& channels);
        virtual const ChannelDesc::Transform& get_transform() const;
    private:
        Channels channels_;
        ChannelDesc::Transform transform_;
        bool has_mask_;
        Texture mask_;
    };
}

