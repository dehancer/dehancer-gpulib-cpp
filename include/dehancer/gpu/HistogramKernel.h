//
// Created by denn on 23.06.2022.
//


#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/histogram.hpp"

namespace dehancer {
    
    namespace impl {
      struct HistogramKernelImpl;
    }
    
    class HistogramKernel: public Kernel {
    
    public:
        using Kernel::Kernel;
    
        explicit HistogramKernel(const void *command_queue,
                                 const Texture &source = nullptr,
                                 bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                                 const std::string &library_path = "");
        
        void setup(CommandEncoder &encoder) override;
        
        const math::Histogram& get_histogram() const;
        
        public:
        std::shared_ptr<impl::HistogramKernelImpl> impl_;
    };
}
