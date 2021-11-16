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
    
    struct FrameSize {
        /***
         * Frame width
         */
        size_t width  = 0;
        
        /***
         * Frame height
         */
        size_t height = 0;
    };
    
    struct Frame {
        FrameSize  size;
        int        count{};
        int        channels{};
        int        channel_depth{};
        float      duration{}; // msec
    };
    
    struct VideoDesc {
        Frame      frame;
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
                mp4
            };
            
            Type type = Options::Type::mp4;
        };
    
    public:
        
        static dehancer::expected<VideoStream,Error> Open(const void *command_queue, const std::string& file_path);
        
        [[nodiscard]] const VideoDesc& get_desc() const;
    
        [[nodiscard]] int get_frame_index() const;
        [[nodiscard]] float get_frame_time() const;
    
        void seek_begin();
        void seek_end();
    
        [[nodiscard]] Texture get_texture_at_time(float time) const;
        [[nodiscard]] Texture get_texture_at_index(int index) const;
        [[nodiscard]] Texture next_texture() const;
        [[nodiscard]] Texture previous_texture() const;
        
        inline static std::string extension_for(Options::Type type) {
          switch (type) {
            case Options::Type::mp4:
              return ".mp4";
          }
        }
    
    private:
        explicit VideoStream(const void *command_queue, const std::string& file_path);
        std::shared_ptr<impl::VideoStream> impl_;
    };
}