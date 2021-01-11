//
// Created by denn nevera on 29/11/2020.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/include/run_test.h"
#include "tests/test_config.h"


namespace test {
    class CustomFilter: public dehancer::Filter {
    public:
        
        explicit CustomFilter(const void* command_queue, const dehancer::Texture& source= nullptr, const dehancer::Texture& destination= nullptr):
        dehancer::Filter(command_queue,source),
        pass_(std::make_shared<dehancer::PassKernel>(command_queue)),
        optic_(std::make_shared<dehancer::OpticalReolution>(command_queue)),
        blur_(std::make_shared<dehancer::GaussianBlur>(command_queue))
        {
          add(pass_, true)
                  .add(optic_, true)
                  .add(blur_, true);
          
          optic_->set_radius(3);
          blur_->set_radius(21.2);
        }
    
        void set_radius(float radius) { radius_ = radius; };
        [[nodiscard]] float get_radius() const {return radius_;};
        
    protected:
        std::shared_ptr<dehancer::PassKernel> pass_;
        std::shared_ptr<dehancer::OpticalReolution> optic_;
        std::shared_ptr<dehancer::GaussianBlur> blur_;

    private:
        
        float radius_ = 0;
    };
}

auto filter_test =  [] (int dev_num,
                        const void* command_queue,
                        const std::string& platform,
                        const std::string& input_image,
                        const std::string& output_image,
                        int image_index) {
    
    auto filter = test::CustomFilter(command_queue);
    
    std::cout << "Load file: " << input_image << std::endl;
    
    /***
     * Load image to texture
     */
    auto input_text = dehancer::TextureInput(command_queue);
    std::ifstream ifs(input_image, std::ios::binary);
    ifs >> input_text;
    
    auto output_text = dehancer::TextureOutput(command_queue, input_text.get_texture(), {
            .type = test::type,
            .compression = test::compression
    });
    
    filter.set_source(input_text.get_texture());
    filter.set_destination(output_text.get_texture());
    
    filter.set_enabling_at(0, false);
    filter.set_enabling_at(1, false);
    
    filter.process(true);
    
    {
      std::ofstream os(output_image, std::ostream::binary | std::ostream::trunc);
      if (os.is_open()) {
        os << output_text << std::flush;
        
        std::cout << "Save to: " << output_image << std::endl;
        
      } else {
        std::cerr << "File: " << output_image << " could not been opened..." << std::endl;
      }
    }
    
    return 0;
};
