//
// Created by denn nevera on 04/10/2020.
//

#pragma once

#include <string>
#include <memory>
#include <functional>

#include "dehancer/gpu/Texture.h"
#include "dehancer/gpu/Memory.h"

namespace dehancer {

    namespace impl { class Command; }

    /***
     * GPU Command layer. Command is the base item of computation.
     * Any command object can be put into device command queue to execute.
     */
    class Command: public std::enable_shared_from_this<Command> {

    public:
        
        static TextureDesc::PixelFormat pixel_format_1d;
        static TextureDesc::PixelFormat pixel_format_2d;
        static TextureDesc::PixelFormat pixel_format_3d;
    
        /***
         * Global property defines waiting or not every command execution
         */
        static bool WAIT_UNTIL_COMPLETED;

        /***
         * Create command from command queue or context. Depends what hardware layer is in base.
         * @param command_queue - command queue or context
         * @param wait_until_completed - flags that can be set to ask wait for ending command execution
         * within current thread or this one should not be locked
         */
        explicit Command(const void *command_queue, bool wait_until_completed = WAIT_UNTIL_COMPLETED);

        /***
         * Make new empty texture binding with the command layer.
         * This version supports float32 texture object with RGBA pixel packing.
         *
         * @param width - texture width
         * @param height - texture height
         * @param depth - texture depth
         * @return - texture object
         */
        Texture make_texture(size_t width, size_t height, size_t depth = 1);

        /***
         * Set waiting flag state
         * @param enable
         */
        virtual void enable_wait_completed(bool enable);

        /***
         * Get current waiting state
         * @return state
         */
        virtual bool get_wait_completed();
    
        virtual void set_wait_completed(bool enable);
        
        /***
         * Get pointer to the command queue descriptor or context handler binds with hardware
         * @return pointer to hardware depended descriptor
         */
        [[nodiscard]] const void* get_command_queue() const;

        /***
         * Get pointer to the command queue descriptor or context handler binds with hardware
         * @return pointer to hardware depended descriptor
         */
        void* get_command_queue();
    
        /**
         * Get self command pointer
         * @return
         */
        std::shared_ptr<Command> get_ptr() { return shared_from_this(); }
    
        virtual TextureInfo get_texture_info(TextureDesc::Type texture_type) const;
        
        virtual ~Command();

    protected:
        std::shared_ptr<impl::Command> impl_;
    };

}

