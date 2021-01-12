//
// Created by denn nevera on 29/11/2020.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/include/run_test.h"
#include "tests/test_config.h"


namespace test {
    
    class Custom3DLut: public dehancer::Kernel {
    public:
        explicit Custom3DLut(const void* command_queue):dehancer::Kernel(command_queue, "kernel_make3DLut_transform"){
          /***
          * Make empty 3D Lut
          */
          clut_ = dehancer::TextureDesc{
                  .width = 64,
                  .height= 64,
                  .depth = 64,
                  .type  = dehancer::TextureDesc::Type::i3d
          }.make(command_queue);
        }
        
        void process() override {
          execute([this](dehancer::CommandEncoder &command_encoder) {
              command_encoder.set(clut_, 0);
              command_encoder.set((dehancer::math::float2) {1, 0}, 1);
              return dehancer::CommandEncoder::Size::From(clut_);
          });
        }
        
        dehancer::Texture get_lut() const { return clut_;}
    
    private:
        dehancer::Texture clut_;
    };
    
    class Custom1DLut: public dehancer::Kernel {
    public:
        explicit Custom1DLut(const void* command_queue):dehancer::Kernel(command_queue, "kernel_make1DLut_transform"){
          /***
          * Make empty 1D curve
          */
          clut_ = dehancer::TextureDesc{
                  .width = 256,
                  .height= 1,
                  .depth = 1,
                  .type  = dehancer::TextureDesc::Type::i1d
          }.make(command_queue);
        }
        
        void process() override {
          execute([this](dehancer::CommandEncoder &command_encoder) {
              command_encoder.set(clut_, 0);
              command_encoder.set((dehancer::math::float2) {0.5, 0.5}, 1);
              return dehancer::CommandEncoder::Size::From(clut_);
          });
        }
        
        dehancer::Texture get_lut() const { return clut_;}
    
    private:
        dehancer::Texture clut_;
    };
    
    class CustomLutTransform: public dehancer::Kernel {
    public:
        explicit CustomLutTransform(const void* command_queue):dehancer::Kernel(command_queue, "kernel_test_transform"){
        }
        
        void setup(dehancer::CommandEncoder &encode) override {
          encode.set(clut_, 2);
          encode.set(clut_curve_, 3);
        }
        
        void set_3d_lut(const dehancer::Texture& clut) { clut_ = clut; }
        void set_curve(const dehancer::Texture& clut_curve) { clut_curve_ = clut_curve; }
    
    private:
        dehancer::Texture clut_;
        dehancer::Texture clut_curve_;
    };
    
    class CustomTransform: public dehancer::Filter {
    public:
        
        explicit CustomTransform(const void* command_queue, const dehancer::Texture& source= nullptr, const dehancer::Texture& destination= nullptr):
                dehancer::Filter(command_queue, source, destination),
                lut3d_transform_(command_queue),
                lut1d_transform_(command_queue),
                trasnform_(std::make_shared<CustomLutTransform>(command_queue))
        {
          
          lut1d_transform_.process();
          lut3d_transform_.process();
          
          trasnform_->set_3d_lut(lut3d_transform_.get_lut());
          trasnform_->set_curve(lut1d_transform_.get_lut());
          
          add(trasnform_);
        }
        
        
    protected:
        Custom3DLut lut3d_transform_;
        Custom1DLut lut1d_transform_;
        std::shared_ptr<CustomLutTransform> trasnform_;
    };
    
    class CustomFilter: public dehancer::Filter {
    public:
        
        explicit CustomFilter(const void* command_queue, const dehancer::Texture& source= nullptr, const dehancer::Texture& destination= nullptr):
                dehancer::Filter(command_queue, source, destination),
                pass_(std::make_shared<dehancer::PassKernel>(command_queue)),
                optic_(std::make_shared<dehancer::OpticalReolution>(command_queue)),
                blur_(std::make_shared<dehancer::GaussianBlur>(command_queue)),
                transform_(std::make_shared<CustomTransform>(command_queue))
        {
          add(pass_, true)
                  .add(optic_, true)
                  .add(blur_, true)
                  .add(transform_);
        }
        
        Filter & process(bool emplace) override {
          std::cout << "pass_ enable: " << is_enable(pass_) << std::endl;
          std::cout << "optic_ enable: " << is_enable(optic_) << std::endl;
          std::cout << "blur_ enable: " << is_enable(blur_) << std::endl;
          std::cout << "transform_ enable: " << is_enable(transform_) << std::endl;
          return dehancer::Filter::process(emplace);
        }
        
        Filter & process() override {
          return process(false);
        }
        
        void set_radius(float radius) {
          radius_ = radius;
          optic_->set_radius(radius_);
          blur_->set_radius(radius_*5.4f);
        };
        
        [[nodiscard]] float get_radius() const {return radius_;};
    
    protected:
        std::shared_ptr<dehancer::PassKernel> pass_;
        std::shared_ptr<dehancer::OpticalReolution> optic_;
        std::shared_ptr<dehancer::GaussianBlur> blur_;
        std::shared_ptr<CustomTransform> transform_;
    
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
    
    filter.set_radius(3);
    
    filter.set_enable(0, false);
    filter.set_enable(1, true);
    filter.set_enable(3, true);
    
    filter.process() ;
    
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
