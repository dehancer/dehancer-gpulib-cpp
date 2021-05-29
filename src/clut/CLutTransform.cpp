//
// Created by denn on 25.04.2021.
//

#include "dehancer/gpu/clut/CLutTransform.h"
#include "dehancer/gpu/clut/CLut1DIdentity.h"
#include "dehancer/gpu/clut/CLut2DIdentity.h"
#include "dehancer/gpu/clut/CLut3DIdentity.h"
#include "dehancer/gpu/clut/CLutHaldIdentity.h"

namespace dehancer {
    
    CLutTransform::CLutTransform(const void *command_queue,
                                 const CLut &lut,
                                 Type to,
                                 const StreamSpace &space,
                                 StreamSpaceDirection direction,
                                 bool wait_until_completed,
                                 const std::string &library_path) :
            CLut(),
            space_(space),
            direction_(direction),
            clut_(nullptr),
            lut_size_(lut.get_lut_size()),
            type_(to)
    {
      if(initializer(command_queue,lut,to)) {
        CLutTransformFunction(
                command_queue,
                kernel_name_,
                lut.get_texture(),
                clut_->get_texture(),
                true,
                library_path
        );
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
              clut_ = std::make_shared<CLut2DIdentity>(command_queue, 64);
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

//        case Type::lut_hald:
//
//          tmp_lut_ = std::make_shared<CLut3DLutFromHaldLut>(command_queue, lut, true);
//          input_texture_ = tmp_lut_->get_texture();
//
//          switch (to) {
//            case Type::lut_3d:
//            case Type::lut_hald:
//              kernel_name_ = "kernel_resample3DLut_to_3DLut";
//              //cs_kernel_name_ = "kernel_cs_transform_3DLut";
//              identity_lut_ = std::make_shared<CLut3DIdentity>(command_queue, lut_size_);
//              texture_ = identity_lut_->get_texture();
//
//              break;
//
//            case Type::lut_2d:
//              kernel_name_ = "kernel_convert3DLut_to_2DLut";
//              //cs_kernel_name_ = "kernel_cs_transform_2DLut";
//              identity_lut_ = std::make_shared<CLut2DIdentity>(command_queue, lut_size_);
//              texture_ = identity_lut_->get_texture();
//
//              break;
//
//            case Type::lut_1d:
//              kernel_name_ = "kernel_convert3DLut_to_1DLut";
//              //cs_kernel_name_ = "kernel_cs_transform_1DLut";
//              identity_lut_ = std::make_shared<CLut1DIdentity>(command_queue, lut_size_);
//              texture_ = identity_lut_->get_texture();
//              break;
//          }
//          break;
      }
  
      std::cerr << " ### CLutTransform::initializer: source clut: " << (int)lut.get_lut_type() << " size: " << lut.get_lut_size() << std::endl;
      std::cerr << " ### CLutTransform::initializer: target clut: " << (int)clut_->get_lut_type() << " size: " << clut_->get_lut_size() << std::endl;
  
      return clut_->get_texture() != nullptr;
    }
}