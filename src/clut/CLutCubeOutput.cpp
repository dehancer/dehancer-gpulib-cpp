//
// Created by denn nevera on 21/05/2020.
//

#include "dehancer/gpu/clut/CLutCubeOutput.h"
#include "dehancer/gpu/clut/CLutTransform.h"
#include "dehancer/gpu/clut/utils/CLut3DCopyFunction.h"
#include "dehancer/vectors.hpp"
#include "gpulib_version.h"

namespace dehancer {
    
    /// Lut export generator version
    
    static std::string generator_comment =  "# Dehancer Core Generator v.";
    
    CLutCubeOutput::CLutCubeOutput(const void *command_queue,
                                   const CLut &clut,
                                   const Options &options,
                                   const std::string &title,
                                   const std::string &comments) :
            command_queue_(command_queue),
            lut_(new CLutTransform(command_queue_,
                                   clut,
                                   CLut::Type::lut_3d,
                                   dehancer::stream_space_identity(),
                                   DHCR_None)),
            title_(title),
            comments_(comments),
            resolution_(options.get_resolution_size())
    {
      std::cerr << " ### CLutCubeOutput::initializer: target clut: " << (int)lut_->get_lut_type() << " size: " << lut_->get_lut_size() << std::endl;
    }
    
    std::ostream &operator<<(std::ostream &os, const CLutCubeOutput &dt) {
      
      auto texture_ = dt.lut_->get_texture();
      
      auto _3dcopy = CLut3DCopyFunction(dt.command_queue_, texture_, dt.lut_->get_lut_size(), true);
      
      os << generator_comment << dehancer::gpulib::version();
      os << std::endl;
      os << std::endl;
      os << "TITLE \""<<dt.title_<<"\"";
      os << std::endl;
      
      if (!dt.comments_.empty()) {
        os << std::endl;
        os << dt.comments_;
        os << std::endl;
        os << std::endl;
      }
      
      os << "# LUT SIZE " << _3dcopy.get_lut_size() << "x" << _3dcopy.get_lut_size() << "x" << _3dcopy.get_lut_size();
      os << ", " << _3dcopy.get_bytes_per_image() << ", " << _3dcopy.get_image_bytes();
      os << std::endl;
      os << "LUT_3D_SIZE " << _3dcopy.get_lut_size();
      os << std::endl;
      os << std::endl;
      os << "# Data domain";
      os << std::endl;
      os << "DOMAIN_MIN 0 0 0";
      os << std::endl;
      os << "DOMAIN_MAX 1 1 1";
      os << std::endl;
      os << std::endl;
      os << "# LUT data points begin";
      os << std::endl;
      
      _3dcopy.foreach([&os](uint index, float r, float g, float b){
          os << std::fixed << std::setw( 1 ) << std::setprecision( 6 )
             << r << " "
             << g << " "
             << b << std::endl;
      });
      
      os << "#LUT data points end" << std::endl;
      
      return os;
    }
    
    CLutCubeOutput::CLutCubeOutput(
            const void *command_queue,
            const CLut &clut,
            size_t resolution,
            const std::string &title,
            const std::string &comments) :
            command_queue_(command_queue),
            lut_(new CLutTransform(command_queue_,
                                   clut,
                                   CLut::Type::lut_3d,
                                   dehancer::stream_space_identity(),
                                   DHCR_None)),
            title_(title),
            comments_(comments),
            resolution_(resolution) {
      
    }
}