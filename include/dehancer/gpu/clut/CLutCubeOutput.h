//
// Created by denn nevera on 21/05/2020.
//

#pragma once

#include "dehancer/gpu/clut/CLut.h"

#include <vector>
#include <cinttypes>
#include <iostream>

namespace dehancer {

    class CLutCubeOutput {
    public:
        static std::string generator_comment;

    public:

        struct Options {
            enum Resolution:int {
                small  = 0,
                normal = 1,
                large  = 2
            };

            Resolution resolution = Resolution::normal;

            [[nodiscard]] size_t get_resolution_size() const  {
                switch (resolution) {
                    case small:
                        return 17;
                    case normal:
                        return 33;
                    case large:
                        return 65;
                }
            }
        };

    public:
        CLutCubeOutput(const void *command_queue,
                       const CLut &clut,
                       const Options &options = {
                               .resolution =  Options::Resolution::normal
                       },
                       const std::string &title = "Dehancer Cube Look Up Table",
                       const std::string &comments = "");

        CLutCubeOutput(const void *command_queue,
                       const CLut &clut,
                       size_t resolution ,
                       const std::string &title = "Dehancer Cube Look Up Table",
                       const std::string &comments = "");

        friend std::ostream& operator<<(std::ostream& os, const CLutCubeOutput& dt);

    private:
        const void* command_queue_;
        std::shared_ptr<CLut> lut_;
        std::string title_;
        std::string comments_;
        size_t      resolution_;
    };
}