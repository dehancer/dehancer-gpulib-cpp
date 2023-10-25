//
// Created by denn nevera on 2019-08-02.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include "dehancer/gpu/kernels/types.h"

namespace dehancer {

    /**
     * Resample kernel
     */
    class ResizeKernel: public Kernel {

    public:
        using Kernel::Kernel;

        /***
         * Resize mode
         */
        enum Mode {
            gauss = 0,
            lanczos = 1
        };

        explicit ResizeKernel(const void *command_queue,
                              const Texture &source,
                              const Texture &destination,
                              Mode mode = gauss,
                              float radius = 1.0f,
                              bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                              const std::string &library_path = "");

        explicit ResizeKernel(const void *command_queue,
                              Mode mode = gauss,
                              float radius = 1.0f,
                              bool wait_until_completed = WAIT_UNTIL_COMPLETED,
                              const std::string &library_path = "");

        void process() override;
        void process(const Texture& source, const Texture& destination) override;

        void set_radius(float radius) { radius_ = radius;};
        float get_radius() const { return radius_;};

        void set_mode(Mode mode) { mode_ = mode; }
        Mode get_mode() const { return mode_; };

    private:
        float radius_;
        Mode mode_;
    };
}