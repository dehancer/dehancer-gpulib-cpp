//
// Created by denn nevera on 2019-10-23.
//

#include "dehancer/gpu/spaces/StreamSpaceCache.h"
#include "dehancer/Log.h"

namespace dehancer {
    
    template<> stream_space_cache * ControlledSingleton<stream_space_cache>::instance = nullptr;

    void stream_space_cache::invalidate() {
      std::unique_lock<std::mutex> lock(mutex_);
      clut_cache_.clear();
    }
    
    size_t stream_space_cache::get_hash(void *command_queue,
                                        const StreamSpace& space,
                                        StreamSpaceDirection direction) {
      return
              std::hash<std::string>()(space.id)
              + reinterpret_cast<std::size_t>(command_queue)
              + std::hash<size_t>()(static_cast<size_t>(direction))                                   * 100
              + std::hash<size_t>()(static_cast<size_t>(space.type))                                  * 10000
              + std::hash<size_t>()(static_cast<size_t>(space.expandable))                            * 1000000
              + std::hash<size_t>()(static_cast<size_t>(space.transform_func.cs_params.log.enabled))  * 100000000
              + std::hash<size_t>()(static_cast<size_t>(space.transform_func.cs_params.gamma.enabled)) * 1000000000
              + std::hash<size_t>()(static_cast<size_t>(space.transform_func.cs_params.log.base * 100.f))  * 10000
              + std::hash<size_t>()(static_cast<size_t>(space.transform_func.cs_params.gamma.base * 100.0f)) * 10000
              ;
      
    }
    
    std::shared_ptr<CLut> stream_space_cache::get_lut(
            void *command_queue,
            const StreamSpace& space,
            StreamSpaceDirection direction) {
      
      const DHCR_LutParameters* transform_lut = nullptr;
      
      switch (direction) {
        
        case StreamSpaceDirection::DHCR_Forward:
          
          if (space.transform_lut.forward.enabled)
            transform_lut = &space.transform_lut.forward;
          break;
        
          case StreamSpaceDirection::DHCR_Inverse:
          
          if (space.transform_lut.inverse.enabled)
            transform_lut = &space.transform_lut.inverse;
          break;
          
        case StreamSpaceDirection::DHCR_None:
          transform_lut = nullptr;
          break;
      }
      
      std::unique_lock<std::mutex> lock(mutex_);
      
      if (!transform_lut) {
        static DHCR_StreamSpace_TransformLut  lut  = stream_space_transform_lut_identity();
        transform_lut = &lut.forward;
      }
      
      size_t hash = get_hash(command_queue,space,direction);
      
      const auto it = clut_cache_.find(hash);
      
      if (it == clut_cache_.end())
      {
        
        auto clut = std::make_shared<CLutCubeInput>(command_queue);
        
        auto e = clut->
                load_from_data(
                transform_lut->data,
                transform_lut->size);
  
        if (e) {
          return nullptr;
        }
        
        clut_cache_[hash] = clut;
        
        return clut;
      }
      else {
        
        return  it->second;
      }
    }
}
