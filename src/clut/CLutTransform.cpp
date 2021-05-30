//
// Created by denn on 25.04.2021.
//

#include "dehancer/gpu/clut/CLutTransform.h"
#include "dehancer/gpu/clut/CLut1DIdentity.h"
#include "dehancer/gpu/clut/CLut2DIdentity.h"
#include "dehancer/gpu/clut/CLut3DIdentity.h"
#include "dehancer/gpu/clut/CLutHaldIdentity.h"

namespace dehancer {
    
    class CLutHaldTo3DTransform : public CLut {
    public:
        CLutHaldTo3DTransform(const void *command_queue,
                              const CLut &lut):
                CLut(),
                texture_(nullptr),
                lut_size_(lut.get_lut_size())
        {
          TextureDesc desc = {
                  .width = lut_size_,
                  .height = lut_size_,
                  .depth = lut_size_,
                  .pixel_format = TextureDesc::PixelFormat::rgba32float,
                  .type = TextureDesc::Type::i3d,
                  .mem_flags = TextureDesc::MemFlags::read_write
          };
          std::vector<float> buffer;
          lut.get_texture()->get_contents(buffer);
          texture_ = desc.make(command_queue,buffer.data());
        }
        
        const Texture& get_texture() override { return texture_; };
        [[nodiscard]] const Texture& get_texture() const override { return texture_; };
        [[nodiscard]] size_t get_lut_size() const override { return lut_size_; };
        [[nodiscard]] Type get_lut_type() const override { return CLut::Type::lut_3d; };
    
    private:
        Texture texture_;
        size_t  lut_size_;
    };
    
    CLutTransform::CLutTransform (const void *command_queue,
                                  const CLut &lut,
                                  CLut::Type to,
                                  size_t lut_size,
                                  const StreamSpace &space,
                                  StreamSpaceDirection direction,
                                  bool wait_until_completed,
                                  const std::string &library_path) :
            CLut(),
            space_(space),
            direction_(direction),
            clut_(nullptr),
            tmp_lut_(nullptr),
            lut_size_(lut_size == 0 ? lut.get_lut_size() : lut_size),
            type_(to)
    {
      if(initializer(command_queue,lut,to)) {
        CLutTransformFunction(
                command_queue,
                kernel_name_,
                tmp_lut_ ? tmp_lut_->get_texture() : lut.get_texture(),
                clut_->get_texture(),
                true,
                library_path
        );
        tmp_lut_ = nullptr;
      }
    }
    
    bool CLutTransform::initializer (const void *command_queue, const CLut &lut, CLut::Type to) {
      kernel_name_ = "kernel_passthrough";
      
      switch (lut.get_lut_type()) {
        
        case Type::lut_1d:
          
          switch (to) {
            case Type::lut_3d:
              kernel_name_ = "kernel_convert1DLut_to_3DLut";
              clut_ = std::make_shared<CLut3DIdentity>(command_queue, lut_size_);
              break;
            
            case Type::lut_2d:
              kernel_name_ = "kernel_convert1DLut_to_2DLut";
              clut_ = std::make_shared<CLut2DIdentity>(command_queue, lut_size_);
              break;
            
            default:
              kernel_name_ = "kernel_resample3DLut_to_3DLut";
              clut_ = std::make_shared<CLut1DIdentity>(command_queue, lut_size_);
              break;
          }
          break;
        
        case Type::lut_2d:
          
          switch (to) {
            case Type::lut_3d:
              kernel_name_ = "kernel_convert2DLut_to_3DLut";
              clut_ = std::make_shared<CLut3DIdentity>(command_queue, lut_size_);
              break;
            
            case Type::lut_1d:
              kernel_name_ = "kernel_convert2DLut_to_1DLut";
              clut_ = std::make_shared<CLut1DIdentity>(command_queue, lut_size_);
              break;
            
            default:
              kernel_name_ = "kernel_resample2DLut_to_2DLut";
              clut_ = std::make_shared<CLut2DIdentity>(command_queue, lut_size_);
              break;
          }
          break;
        
        case Type::lut_3d:
          
          switch (to) {
            case Type::lut_2d:
              kernel_name_ = "kernel_convert3DLut_to_2DLut";
              clut_ = std::make_shared<CLut2DIdentity>(command_queue, lut_size_);
              break;
            
            case Type::lut_1d:
              kernel_name_ = "kernel_convert3DLut_to_1DLut";
              clut_ = std::make_shared<CLut1DIdentity>(command_queue, lut_size_);
              break;
            
            default:
              kernel_name_ = "kernel_resample3DLut_to_3DLut";
              clut_ = std::make_shared<CLut3DIdentity>(command_queue, lut_size_);
              std::cerr << " ### CLutTransform::initializer: >>: " << kernel_name_ << std::endl;
              
              break;
          }
          break;
        
        case Type::lut_hald:
          
          tmp_lut_ = std::make_shared<CLutHaldTo3DTransform>(command_queue, lut);
          
          switch (to) {
            case Type::lut_3d:
              kernel_name_ = "kernel_resample3DLut_to_3DLut";
              clut_ = std::make_shared<CLut3DIdentity>(command_queue, lut_size_);
              break;
            case Type::lut_2d:
              kernel_name_ = "kernel_convert3DLut_to_2DLut";
              clut_ = std::make_shared<CLut2DIdentity>(command_queue, lut_size_);
              break;
            
            case Type::lut_1d:
              kernel_name_ = "kernel_convert3DLut_to_1DLut";
              clut_ = std::make_shared<CLut1DIdentity>(command_queue, lut_size_);
              break;
          }
          break;
      }
      std::cerr << " ### CLutTransform::initializer: source clut: " << (int)lut.get_lut_type() << " size: " << lut.get_lut_size() << std::endl;
      std::cerr << " ### CLutTransform::initializer: target clut: " << (int)clut_->get_lut_type() << " size: " << clut_->get_lut_size() << std::endl;
      
      return clut_->get_texture() != nullptr;
    }
}