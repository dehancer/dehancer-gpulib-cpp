//
// Created by denn nevera on 05/06/2020.
//

#pragma once

#include "dehancer/Common.h"
#include "dehancer/gpu/Texture.h"

namespace dehancer {
    
    namespace impl {
        struct VideoStream;
    }
    
    struct KeyFrame {
    
        struct Size {
            /***
             * Frame width
             */
            size_t width  = 0;
        
            /***
             * Frame height
             */
            size_t height = 0;
        };
        
        Size       size;
        int        count{};
        int        channels{};
        int        channel_depth{};
        float      duration{}; // msec
    };
    
    struct VideoDesc {
        KeyFrame   keyframe;
        int        type{};
        float      time{};    // msec
        float      bitrate{}; //
        float      fps{};
    };
    
    
    /***
    * Texture Input/Output interface
    */
    class VideoStream {
    public:
        
        struct Options {
            
            enum Type {
                mp4,
                mov
            };
            
            Type type = Options::Type::mp4;
        };
    
    public:
        
        static dehancer::expected<VideoStream,Error> Open(
                const void *command_queue,
                const std::string& file_path);
        
        explicit VideoStream(const void *command_queue,
                             const std::string& file_path);
    
        [[nodiscard]] const VideoDesc& get_desc() const;
    
        [[nodiscard]] int   current_keyframe_position() const;
        
        /**
         *
         * @return current stream position at time in milliseconds
         */
        [[nodiscard]] float current_keyframe_time() const;
    
        void seek_at_time(float time);
        void skip_backward();
        void skip_forward();
    
        /***
         *
         * @param time in milliseconds
         * @return current texture
         */
        [[nodiscard]] Texture get_texture_at_time(float time) const;
        [[nodiscard]] Texture get_texture_in_keyframe(int position) const;
        [[nodiscard]] Texture next_texture() const;
        [[nodiscard]] Texture previous_texture() const;
        
        inline static std::string extension_for(Options::Type type) {
          switch (type) {
            case Options::Type::mp4:
              return ".mp4";
            case Options::Type::mov:
              return ".mov";
          }
        }
    
    private:
        std::shared_ptr<impl::VideoStream> impl_;
    };
}