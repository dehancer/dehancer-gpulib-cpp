//
// Created by denn nevera on 17/11/2020.
//

#pragma once

#include "dehancer/gpu/Function.h"

namespace dehancer {

    namespace impl { struct Kernel; }

    class Kernel: public Function  {

    public:
        explicit Kernel(
                const void *command_queue,
                const std::string& kernel_name,
                const Texture& source,
                const Texture& destination,
                bool wait_until_completed = WAIT_UNTIL_COMPLETED
        );

        FunctionHandler optionsHandler = nullptr;

        virtual void process();

        virtual void setup(CommandEncoder &encode);

        [[nodiscard]] virtual const Texture& get_source() const;
        [[nodiscard]] virtual const Texture& get_destination() const;
        virtual void set_destination(Texture& dest);

        ~Kernel() override;

    private:
        std::shared_ptr<impl::Kernel> impl_;
    };

}


