//
// Created by denn on 25.04.2021.
//

#pragma once

#include "dehancer/gpu/Function.h"
#include "dehancer/gpu/clut/CLutTransformFunction.h"
#include "dehancer/gpu/clut/CLut.h"
#include "dehancer/gpu/StreamSpace.h"

namespace dehancer {
    
    class CLutTransform: public CLut {
    
    public:
        /***
         *
         * Transform CLut object from one dimension to another
         * @param command_queue - command queue
         * @param lut - source lut object
         * @param to - destination lut type
         * TODO:
         * @param space - not works yet
         * @param direction - not works yet
         */
        CLutTransform(const void *command_queue,
                      const CLut &lut,
                      CLut::Type to,
                      const StreamSpace &space = stream_space_identity(),
                      StreamSpaceDirection direction = StreamSpaceDirection::DHCR_None,
                      bool wait_until_completed = Function::WAIT_UNTIL_COMPLETED,
                      const std::string &library_path = "");
        
        const Texture& get_texture() override { return clut_->get_texture(); };
        [[nodiscard]] const Texture& get_texture() const override { return clut_->get_texture(); };
        [[nodiscard]] size_t get_lut_size() const override { return lut_size_; };
        [[nodiscard]] Type get_lut_type() const override { return type_; };
        
        ~CLutTransform() override = default;

    protected:
        const StreamSpace&  space_;
        const StreamSpaceDirection direction_;

    private:
        bool initializer(const void *command_queue, const CLut &lut, Type to);
    
        //Texture input_texture_;
        std::shared_ptr<CLut> clut_;
        size_t  lut_size_;
        Type    type_;
        std::string kernel_name_;
    };
    
}