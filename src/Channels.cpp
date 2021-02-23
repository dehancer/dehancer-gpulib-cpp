//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/Channels.h"
#include "dehancer/gpu/Log.h"
#include "cache/cache.hpp"
#include "cache/lru_cache_policy.hpp"
#include <cmath>

namespace dehancer {
    
    static dehancer::ChannelsDesc::Transform options_one = {
            .slope = {8.0f,4,0,0},
            .offset = {128.0f,128,0,0},
            .enabled = {true,true,false,false},
            .direction = dehancer::ChannelsDesc::TransformDirection::forward,
            .mask = nullptr
    };
    
    ChannelsDesc::Scale2D ChannelsDesc::default_scale = {(Scale){1.0f, 1.0f}, {1.0f,1.0f}, {1.0f,1.0f}, {1.0f,1.0f}};
    
    Channels ChannelsDesc::make (const void *command_queue) const {
      return ChannelsHolder::Make(command_queue, *this);
    }
    
    size_t ChannelsDesc::get_hash () const {
      size_t cx = 0;
      for (int i = 0; i < scale.size(); ++i) {
        cx += static_cast<size_t>( scale[i].x * 10000 + scale[i].y * 100 ) << i;
      }
      return
      1000000000 * width
      +
      10000000 * height
      +
      cx;
    }
    
    namespace impl {
        
        struct ChannelItem {
            size_t                                hash;
            std::shared_ptr<std::array<Memory,4>> channels = std::make_shared<std::array<Memory,4>>();
    
            ~ChannelItem() {
              #ifdef PRINT_DEBUG
              dehancer::log::print(" ### ~ChannelItem(base): %p: %li", this, hash);
              #endif
            }
        };
    
        namespace text {
            template<typename Key, typename Value>
            using lru_cache_t = typename caches::fixed_sized_cache<Key, Value, caches::LRUCachePolicy<Key>>;
        }
    
        using channels_pool_t = std::vector<std::shared_ptr<ChannelItem>>;
    
        using channels_cache_t = text::lru_cache_t<size_t, std::shared_ptr<channels_pool_t>>;
        using device_channels_cache_t = text::lru_cache_t<size_t, std::shared_ptr<channels_cache_t>>;
    
        static device_channels_cache_t device_channels_cache(4);
    
    
        struct ChannelsHolder: public dehancer::ChannelsHolder, public dehancer::Command {
            
            //std::shared_ptr<std::array<Memory,4>> channels_;
            std::shared_ptr<ChannelItem> item_;
            ChannelsDesc desc_;
            std::shared_ptr<std::array<ChannelsDesc,4>> channel_descs_;
            
            size_t get_width(int index) const override { return channel_descs_->at(index).width; };
            size_t get_height(int index) const override {return channel_descs_->at(index).height;};
    
            ChannelsDesc::Scale get_scale(int index) const override { return desc_.scale.at(index);}
            
            ChannelsDesc get_desc() const override { return desc_;}
            
            Memory& at(int index) override { return item_->channels->at(index);};
            const Memory& at(int index) const override { return item_->channels->at(index);};
            [[nodiscard]] size_t size() const override { return item_->channels->size(); };
            
            ChannelsHolder(const void *command_queue, const ChannelsDesc& desc):
                    Command(command_queue),
                    //channels_(std::make_shared<std::array<Memory,4>>()),
                    item_(std::make_shared<ChannelItem>()),
                    desc_(desc),
                    channel_descs_({
                                           std::make_shared<std::array<ChannelsDesc,4>>()
                                   })
            {
  
              auto hash = desc_.get_hash();
              auto dev_hash  = reinterpret_cast<size_t>(command_queue);
              
              int i = 0;
              for(auto& c: *channel_descs_){
                c = desc_;
                c.width = std::floor((float )c.width * c.scale.at(i).x);
                c.height = std::floor((float )c.height * c.scale.at(i).y);
                ++i;
              }
              
              i = 0;
              
              item_->hash = hash;
  
              std::shared_ptr<channels_cache_t> c_cache;
              
              try {
                c_cache = device_channels_cache.Get(dev_hash);
              }
              catch (...) {
                c_cache = std::make_shared<channels_cache_t>(16);
                device_channels_cache.Put(dev_hash,c_cache);
              }
              bool is_cached = false;
  
              if (c_cache->Cached(hash) && !c_cache->Get(hash)->empty()) {
    
                auto& q = c_cache->Get(hash);
                item_ = q->back(); q->pop_back();
  
                is_cached = true;
  
              }
              else {
                for (auto &c : *item_->channels) {
                  auto size = sizeof(float) * channel_descs_->at(i).width * channel_descs_->at(i).height;
                  if (size == 0) continue;
                  c = MemoryHolder::Make(get_command_queue(), size);
                  ++i;
                }
  
                if (!c_cache->Cached(hash))
                  c_cache->Put(hash, std::make_shared<channels_pool_t>());
  
                c_cache->Get(hash)->push_back(item_);
              }
  
              #ifdef PRINT_DEBUG
              dehancer::log::print(" ### %s ChannelsHolder(base): %p: %li  : %ix%i",
                                   is_cached ? "Cached" : "New",
                                   item_->channels.get(), item_->hash,
                                   desc_.width, desc_.height);
              #endif
            }
    
            ~ChannelsHolder() override {
             
              auto hash = desc_.get_hash();
              auto dev_hash  = reinterpret_cast<size_t>(get_command_queue());
  
              std::shared_ptr<channels_cache_t> c_cache;
  
              try {
                c_cache = device_channels_cache.Get(dev_hash);
              }
              catch (...) {
                c_cache = std::make_shared<channels_cache_t>(16);
                device_channels_cache.Put(dev_hash,c_cache);
              }
  
              if (c_cache->Cached(hash) && !c_cache->Get(hash)->empty()) {
  
                #ifdef PRINT_DEBUG
                dehancer::log::print(" ### RETURN ~ChannelsHolder(base): %p: %ix%i", this, desc_.width, desc_.height);
                #endif
  
                c_cache->Get(hash)->push_back(item_);
              }
            }
        };
    }
    
    Channels ChannelsHolder::Make(const void *command_queue,
                                  size_t width,
                                  size_t height) {
      ChannelsDesc desc = {
              .width = width,
              .height = height
      };
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc);
    }
    
    Channels ChannelsHolder::Make (const void *command_queue, const ChannelsDesc &desc) {
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc);
    }
    
    
    ChannelsInput::ChannelsInput(const void *command_queue,
                                 const Texture &texture,
                                 const ChannelsDesc::Transform& transform,
                                 ChannelsDesc::Scale2D scale,
                                 bool wait_until_completed,
                                 const std::string& library_path):
            Kernel(command_queue,
                   "image_to_one_channel",
                   texture,
                   nullptr,
                   wait_until_completed,
                   library_path),
            desc_({
                          .width = texture ? texture->get_width() : 0,
                          .height = texture ? texture->get_height() : 0,
                          .scale = scale
                  }),
            channels_(nullptr),
            has_mask_(transform_.mask != nullptr),
            transform_(transform)
    {
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
    }
    
    void ChannelsInput::process () {
      
      if (!channels_) {
        channels_ = desc_.make(get_command_queue());
      }
      
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      
      for (int j = 0; j < channels->size(); ++j) {
        
        execute([this, channels, j](CommandEncoder& encoder){
            
            encoder.set(get_source(),0);
            
            encoder.set(channels->at(j),1);
            
            int cw = channels->get_width(j);
            int ch = channels->get_height(j);
            
            encoder.set(cw, 2);
            encoder.set(ch, 3);
            
            encoder.set(j, 4);
            
            encoder.set(transform_.slope[j],5);
            encoder.set(transform_.offset[j],6);
            
            if (transform_.flags.in_enabled)
              encoder.set(transform_.enabled[j],7);
            else
              encoder.set(false,7);
    
            encoder.set(transform_.direction ,8);
            encoder.set(transform_.type ,9);
            
            encoder.set(has_mask_ , 10);
            encoder.set(mask_ , 11);
            
            CommandEncoder::Size size = {
                    .width = channels->get_width(j),
                    .height = channels->get_height(j),
                    .depth = 1
            };
            
            return size;
        });
      }
    }
    
    void ChannelsInput::process (const Texture &source, const Texture &destination) {
      Kernel::process(source, destination);
    }
    
    
    void ChannelsInput::set_source (const Texture &source) {
      Kernel::set_source(source);
      if (!source) {
        channels_ = nullptr;
        return;
      }
      
      desc_.width = source->get_width();
      desc_.height = source->get_height();
      
      if (channels_
          &&
          (source->get_width()!=channels_->get_desc().width
           ||
           source->get_height()!=channels_->get_desc().height))
        channels_ = nullptr;
    }
    
    void ChannelsInput::set_destination (const Texture &destination) {
      Kernel::set_destination(nullptr);
    }
    
    void ChannelsInput::set_transform (const ChannelsDesc::Transform &transform) {
      transform_ = transform;
      has_mask_ = transform_.mask != nullptr;
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
      else
        mask_ = transform_.mask;
    }
    
    const ChannelsDesc::Transform &ChannelsInput::get_transform () const {
      return transform_;
    }
    
    void ChannelsInput::set_scale (ChannelsDesc::Scale2D scale) {
      bool do_recreate_channels = false;
      for (int i = 0; i < scale.size(); ++i) {
        if (desc_.scale.at(i).x!=scale.at(i).y || desc_.scale.at(i).y!=scale.at(i).y) {
          do_recreate_channels = true;
          break;
        }
      }
      desc_.scale = scale;
      if (do_recreate_channels){
        channels_ = nullptr;
      }
    }
    
    
    ChannelsOutput::ChannelsOutput(const void *command_queue,
                                   const Texture& destination,
                                   const Channels& channels,
                                   const ChannelsDesc::Transform& transform,
                                   bool wait_until_completed,
                                   const std::string& library_path):
            Kernel(command_queue,
                   "one_channel_to_image",
                   nullptr,
                   destination,
                   wait_until_completed,
                   library_path),
            channels_(channels),
            has_mask_(transform_.mask != nullptr),
            transform_(transform)
    {
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
    }
    
    void ChannelsOutput::process () {
     
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(channels_.get());
      
      
      for (int j = 0; j < channels->size(); ++j) {
        
        auto channel = channels->at(j);
        
        execute([this, channels, &channel, j](CommandEncoder& encoder){
            
            encoder.set(get_destination(),0);
            
            encoder.set(get_destination(),1);
            
            encoder.set(channel,2);
            
            int cw = channels->get_width(j);
            int ch = channels->get_height(j);
            
            encoder.set(cw, 3);
            encoder.set(ch, 4);
            
            encoder.set(j, 5);
            
            encoder.set(transform_.slope[j],6);
            encoder.set(transform_.offset[j],7);
            
            if (transform_.flags.out_enabled)
              encoder.set(transform_.enabled[j],8);
            else
              encoder.set(false,8);
            
            encoder.set(transform_.direction ,9);
    
            encoder.set(transform_.type ,10);
    
            encoder.set(has_mask_ , 11);
            encoder.set(mask_ , 12);
            
            return CommandEncoder::Size::From(get_destination());
        });
      }
    }
    
    void ChannelsOutput::process (const Texture &source, const Texture &destination) {
      Kernel::process(source, destination);
    }
    
    void ChannelsOutput::set_destination (const Texture &destination) {
      Kernel::set_destination(destination);
    }
    
    void ChannelsOutput::set_source (const Texture &source) {
      Kernel::set_source(nullptr);
    }
    
    void ChannelsOutput::set_transform (const ChannelsDesc::Transform &transform) {
      transform_ = transform;
      has_mask_ = transform_.mask != nullptr;
      if (!transform_.mask) {
        TextureDesc desc ={
                .width = 1,
                .height = 1
        };
        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
        mask_ = desc.make(get_command_queue(),mem);
      }
      else {
        mask_ = transform_.mask;
      }
    }
    
    const ChannelsDesc::Transform &ChannelsOutput::get_transform () const {
      return transform_;
    }
    
    void ChannelsOutput::set_channels (const Channels &channels) {
      channels_ = channels;
    }
}