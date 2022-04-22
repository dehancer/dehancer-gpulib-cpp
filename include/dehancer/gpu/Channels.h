//
// Created by denn nevera on 30/11/2020.
//

#pragma once

#include <array>
#include "dehancer/gpu/Memory.h"
#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Kernel.h"

namespace dehancer {
    
    struct ChannelsHolder;
    
    /***
     * Memory pointer object
     */
    using Channels = std::shared_ptr<ChannelsHolder>;
    
    namespace impl {
        struct ChannelsHolder;
    }
    
    /***
     * Channels Description
     */
    struct ChannelsDesc {
    
        using ActiveChannelsMask = std::array<bool,4>;
    
        /***
         * Channel values transformation type
         */
        enum TransformType:int {
            /***
             * direction == DHCR_forward
             *   y = 2^(x*slope)-offset
             *
             * direction == DHCR_reverse
             *   y = (log2(x) + offset) / slope, slope!=0
             */
            log_linear = DHCR_log_linear,
            pow_linear = DHCR_pow_linear
        };
        
        /***
         * Channel transformation direction
         */
        enum TransformDirection:int {
            forward = DHCR_TransformDirection::DHCR_Forward,
            inverse = DHCR_TransformDirection::DHCR_Inverse,
            none    = DHCR_TransformDirection::DHCR_None
        };
        
        struct Scale {
            float x = 1.0f;
            float y = 1.0f;
        };
        
        using Scale2D = std::array<Scale,4>;
        
        /***
         * Transformation description
         */
        struct Transform {
            
            struct Flags {
                bool  in_enabled;
                bool out_enabled;
            };
            
            /***
             * Type
             */
            TransformType            type = log_linear;
            
            /***
             * Channels (must be 4: RGBA) log slope
             */
            dehancer::math::float4  slope = { 0.f,0.f,0.f,0.f };
            /***
             * Channels (must be 4: RGBA) log offset
             */
            dehancer::math::float4 offset = { 1.0f,1.0f,1.0f,1.0f };
            /***
             * Transformation applies for the channels by index
             */
            dehancer::math::bool4 enabled = {false,false,false,false};
            
            /***
             * Direction
             */
            TransformDirection  direction = none;
            
            /***
             * Opacity mask applies for channels transformation
             */
            //Texture              mask = nullptr;
            
            Flags                flags = {
                    .in_enabled = true,
                    .out_enabled = true
            };
            
            /***
             * Make default transformation
             * @param type
             * @param slope
             * @param offset
             * @param direction
             * @param enabled
             * @return
             */
            static Transform make(TransformType type, float slope, float offset, TransformDirection direction, bool enabled = true) {
              return {
                      .type = type,
                      .slope =  { slope,slope,slope,0.f },
                      .offset = { offset,offset,offset,1.f },
                      .enabled = {enabled,enabled,enabled,false},
                      .direction = direction
              };
            };
        };
        
        /***
         * 2D mapping dimension width
         */
        size_t  width     = 0;
        /***
         * 2D mapping dimension height
         */
        size_t  height    = 0;
        
        /***
         * Scale channels instead of origin texture size
         */
        Scale2D scale;
        
        [[nodiscard]] size_t get_hash() const;
        
        Channels make(const void *command_queue, const ActiveChannelsMask& amask) const;
        
        static Scale2D default_scale;
    };
    
    struct ChannelsHolder: public std::enable_shared_from_this<ChannelsHolder> {
    
    public:
        
        static Channels Make(const void *command_queue,
                             size_t width,
                             size_t height,
                             const ChannelsDesc::ActiveChannelsMask& amask = {true,true,true,false});
        static Channels Make(const void *command_queue,
                             const ChannelsDesc& desc,
                             const ChannelsDesc::ActiveChannelsMask& amask = {true,true,true,false}
        );
        
        virtual size_t get_width(int index) const = 0;
        virtual size_t get_height(int index) const = 0;
        
        virtual ChannelsDesc::Scale get_scale(int index) const = 0;
        
        virtual ChannelsDesc get_desc() const = 0;
        
        virtual Memory& at(int index) = 0;
        virtual const Memory& at(int index) const = 0;
        [[nodiscard]] virtual inline size_t size() const = 0;
        
        Channels get_ptr() { return shared_from_this(); }
    
        //virtual void set_active_mask(const Ch::ActiveChannelsMask& amask) = 0 ;
        //[[nodiscard]] virtual const ChannelsHolder::ActiveChannelsMask& get_active_mask() const = 0;
        
        virtual ~ChannelsHolder() = default;
    
    protected:
        ChannelsHolder() = default;
    };
    
    namespace impl {
        struct ChannelsInputImpl;
        struct ChannelsOutputImpl;
    }
    
    class ChannelsInput: public Kernel {
    
    public:
        
        using Kernel::Kernel;
        
        explicit ChannelsInput(const void *command_queue,
                               const Texture& source = nullptr,
                               const ChannelsDesc::Transform& transform = {},
                               ChannelsDesc::Scale2D scale = ChannelsDesc::default_scale,
                               const ChannelsDesc::ActiveChannelsMask& amask = {true,true,true,false},
                               bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                               const std::string& library_path = ""
        );
        
        [[nodiscard]] const Channels& get_channels() const;
        
        void process() override;
        void process(const Texture &source, const Texture &destination) override;
        
        void set_scale(ChannelsDesc::Scale2D scale);
        void set_source(const Texture& source) override;
        void set_destination(const Texture& destination) override;
        
        //void set_active_mask(const ChannelsHolder::ActiveChannelsMask& amask);
        
        virtual void set_transform(const ChannelsDesc::Transform& transform);
        virtual const ChannelsDesc::Transform& get_transform() const;
    
    private:
        std::shared_ptr<impl::ChannelsInputImpl> impl_;
    };
    
    class ChannelsOutput: public Kernel {
    
    public:
        
        using Kernel::Kernel;
        
        explicit ChannelsOutput(const void *command_queue,
                                const Texture& destination,
                                const Channels& channels,
                                const ChannelsDesc::Transform& transform = {},
                                //const ChannelsDesc::ActiveChannelsMask& amask = {true,true,true,false},
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string& library_path = ""
        );
        
        void process() override;
        void process(const Texture &source, const Texture &destination) override;
        
        void set_source(const Texture& source) override;
        void set_destination(const Texture& destination) override;
        virtual void set_transform(const ChannelsDesc::Transform& transform);
        virtual void set_channels(const Channels& channels);
        virtual const ChannelsDesc::Transform& get_transform() const;
        [[nodiscard]] const Channels& get_channels() const;
    
        //void set_active_mask(const ChannelsHolder::ActiveChannelsMask& amask);

    private:
        std::shared_ptr<impl::ChannelsOutputImpl> impl_;
    };
}

