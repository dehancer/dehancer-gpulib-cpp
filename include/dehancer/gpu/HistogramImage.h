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
            float left_trim = 1.0f;
            float right_trim = 1.0f;
        };
        
        struct Options {
            bool  ignore_edges;
            Edges edges;
        };
        
        explicit HistogramImage(const void *command_queue,
                                const Texture &source = nullptr,
                                const Options& options = {
                                        .ignore_edges = false,
                                        .edges = Edges {
                                          .left_trim = 1.0f,
                                          .right_trim = 1.0f
                                        }
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