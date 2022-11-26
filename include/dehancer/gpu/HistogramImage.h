//
// Created by denn on 23.06.2022.
//


#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/histogram.hpp"

namespace dehancer {
    
    namespace impl {
        struct HistogramImpl;
    }
    
    class HistogramImage: public Function {
    
    public:
        using Function::Function;
        
        struct Edges {
            bool  ignore = false;
            float left_trim = 1.0f;
            float right_trim = 1.0f;
        };
    
        struct Transform {
            bool  enabled = false;
            StreamSpace space;
            StreamSpaceDirection direction;
        };
    
        enum class LumaType:int {
            YCbCr = 0,
            YUV = 1,
            Mean = 2
        };
        
        struct Options {
            Edges     edges;
            Transform transform;
            LumaType  luma_type = LumaType::YCbCr;
        };
        
        explicit HistogramImage(const void *command_queue,
                                const Texture &source = nullptr,
                                const Options& options = {
                                        .edges = Edges {
                                                .ignore = false,
                                                .left_trim = 1.0f,
                                                .right_trim = 1.0f
                                        },
                                        .transform = {
                                                .enabled = false
                                        },
                                        .luma_type =  LumaType::YCbCr
                                },
                                bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                const std::string &library_path = "");
        
        /***
         * Set a current image texture
         * @param source
         */
        void set_source(const Texture& source);
        
        /**
         * Get source texture
         * @return texture object
         */
        [[nodiscard]] const Texture& get_source() const;
        
        /***
         * Process Histogram
         * */
        void process();
        
        /**
         * Get the currently processed histogram
         * @return histogram object
         */
        const math::Histogram& get_histogram() const;
        
        void set_options(const Options& options);
        
    public:
        std::shared_ptr<impl::HistogramImpl> impl_;
    };
}
