//
// Created by denn nevera on 30/11/2020.
//

#include "dehancer/gpu/Channels.h"
#include "dehancer/gpu/operations/PassKernel.h"
#include "dehancer/gpu/Log.h"
#include <cmath>

namespace dehancer {
    
    ChannelsDesc::Scale2D ChannelsDesc::default_scale = {(Scale){1.0f, 1.0f}, {1.0f,1.0f}, {1.0f,1.0f}, {1.0f,1.0f}};
    
    Channels ChannelsDesc::make (const void *command_queue, const ChannelsDesc::ActiveChannelsMask& amask) const {
      return ChannelsHolder::Make(command_queue, *this, amask);
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
            size_t                                hash{};
            dehancer::ChannelsDesc::ActiveChannelsMask    amask{};
            std::shared_ptr<std::array<Memory,4>> channels = std::make_shared<std::array<Memory,4>>();
        };
        
        
        struct ChannelsHolder: public dehancer::ChannelsHolder, public dehancer::Command {
            
            std::shared_ptr<ChannelItem> item_;
            ChannelsDesc desc_;
            std::shared_ptr<std::array<ChannelsDesc,4>> channel_descs_;
//
//            void set_active_mask(const ChannelsHolder::ActiveChannelsMask& amask) override {
//              //item_->amask = amask;
//              //init();
//            };
//
//            [[nodiscard]] const ChannelsHolder::ActiveChannelsMask& get_active_mask() const override {
//              return item_->amask;
//            };
            
            size_t get_width(int index) const override { return channel_descs_->at(index).width; };
            size_t get_height(int index) const override {return channel_descs_->at(index).height;};
            
            ChannelsDesc::Scale get_scale(int index) const override { return desc_.scale.at(index);}
            
            ChannelsDesc get_desc() const override { return desc_;}
            
            Memory& at(int index) override {
              return item_->channels->at(index);
            };
            
            const Memory& at(int index) const override { return item_->channels->at(index);};
            [[nodiscard]] size_t size() const override { return item_->channels->size(); };
            
            ChannelsHolder(const void *command_queue, const ChannelsDesc& desc, const ChannelsDesc::ActiveChannelsMask& amask ):
                    Command(command_queue),
                    item_(std::make_shared<ChannelItem>()),
                    desc_(desc),
                    channel_descs_({
                                           std::make_shared<std::array<ChannelsDesc,4>>()
                                   })
            {
              item_->amask = amask;
              init();
            }
            
            void init() {
              int i = 0;
              for(auto& c: *channel_descs_){
                c = desc_;
                c.width = std::floor((float )c.width * c.scale.at(i).x);
                c.height = std::floor((float )c.height * c.scale.at(i).y);
                ++i;
              }
              
              i = 0;
              
              item_->hash = desc_.get_hash();
              
              for (auto &c : *item_->channels) {
                auto size = sizeof(float) * channel_descs_->at(i).width * channel_descs_->at(i).height;
                if (size == 0) continue;
                if (item_->amask[i]) {
                  c = MemoryHolder::Make(get_command_queue(), nullptr, size, MemoryDesc::MemFlags::less_memory);
                }
                else {
                  c = nullptr;
                }
                ++i;
              }
            }
            
            ~ChannelsHolder() override  = default;
        };
    }
    
    Channels ChannelsHolder::Make(const void *command_queue,
                                  size_t width,
                                  size_t height,
                                  const ChannelsDesc::ActiveChannelsMask& amask
    ) {
      ChannelsDesc desc = {
              .width = width,
              .height = height
      };
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc,amask);
    }
    
    Channels ChannelsHolder::Make (const void *command_queue, const ChannelsDesc &desc, const ChannelsDesc::ActiveChannelsMask& amask ) {
      return std::make_shared<impl::ChannelsHolder>(command_queue,desc,amask);
    }
    
    
    /***
     *
     * @param command_queue
     * @param texture
     * @param transform
     * @param scale
     * @param wait_until_completed
     * @param library_path
     */
    
    namespace impl {
        struct ChannelsInputImpl {
            ChannelsDesc desc;
            ChannelsDesc::ActiveChannelsMask amask{};
            Channels channels = nullptr;
            ChannelsDesc::Transform transform;
            //bool has_mask{};
            //Texture mask= nullptr;
        };
    }
    
    ChannelsInput::ChannelsInput(const void *command_queue,
                                 const Texture &texture,
                                 const ChannelsDesc::Transform& transform,
                                 ChannelsDesc::Scale2D scale,
                                 const ChannelsDesc::ActiveChannelsMask& amask,
                                 bool wait_until_completed,
                                 const std::string& library_path):
            Kernel(command_queue,
                   "image_to_one_channel",
                   texture,
                   nullptr,
                   wait_until_completed,
                   library_path),
            impl_(std::make_shared<impl::ChannelsInputImpl>())
    {
      
      impl_->desc = {
              .width = texture ? texture->get_width() : 0,
              .height = texture ? texture->get_height() : 0,
              .scale = scale
      };
      
      impl_->amask = amask;
      
      impl_->transform = transform;
      //impl_->has_mask = transform.mask != nullptr;
      
//      if (!impl_->transform.mask) {
//        TextureDesc desc ={
//                .width = 1,
//                .height = 1
//        };
//        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
//        impl_->mask = desc.make(get_command_queue(),mem);
//      }
    }
    
    void ChannelsInput::process () {
      
      if (!impl_->channels) {
        impl_->channels = impl_->desc.make(get_command_queue(), impl_->amask);
      }
      
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(impl_->channels.get());
      
      for (int j = 0; j < channels->size(); ++j) {
        
        auto mem = channels->at(j);
        
        if (!mem) continue;
        
        execute([this, channels, j, &mem](CommandEncoder& encoder){
            
            encoder.set(get_source(),0);
            
            encoder.set(mem,1);
            
            int cw = static_cast<int>(channels->get_width(j));
            int ch = static_cast<int>(channels->get_height(j));
            
            encoder.set(cw, 2);
            encoder.set(ch, 3);
            
            encoder.set(j, 4);
            
            encoder.set(impl_->transform.slope[j],5);
            encoder.set(impl_->transform.offset[j],6);
            
            if (impl_->transform.flags.in_enabled)
              encoder.set(impl_->transform.enabled[j],7);
            else
              encoder.set(false,7);
            
            encoder.set(impl_->transform.direction ,8);
            encoder.set(impl_->transform.type ,9);
            
            //encoder.set(impl_->has_mask , 10);
            //encoder.set(impl_->mask , 11);
            
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
        impl_->channels = nullptr;
        return;
      }
      
      impl_->desc.width = source->get_width();
      impl_->desc.height = source->get_height();
      
      if (impl_->channels
          &&
          (source->get_width()!=impl_->channels->get_desc().width
           ||
           source->get_height()!=impl_->channels->get_desc().height))
        impl_->channels = nullptr;
    }
    
    void ChannelsInput::set_destination (const Texture &destination) {
      Kernel::set_destination(nullptr);
    }
    
    void ChannelsInput::set_transform (const ChannelsDesc::Transform &transform) {
      impl_->transform = transform;
//      impl_->has_mask = transform.mask != nullptr;
//      if (!impl_->transform.mask) {
//        TextureDesc desc ={
//                .width = 1,
//                .height = 1
//        };
//        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
//        impl_->mask = desc.make(get_command_queue(),mem);
//      }
//      else
//        impl_->mask = impl_->transform.mask;
    }
    
    const ChannelsDesc::Transform &ChannelsInput::get_transform () const {
      return impl_->transform;
    }
    
    void ChannelsInput::set_scale (ChannelsDesc::Scale2D scale) {
      bool do_recreate_channels = false;
      for (int i = 0; i < scale.size(); ++i) {
        if (impl_->desc.scale.at(i).x!=scale.at(i).y || impl_->desc.scale.at(i).y!=scale.at(i).y) {
          do_recreate_channels = true;
          break;
        }
      }
      impl_->desc.scale = scale;
      if (do_recreate_channels){
        impl_->channels = nullptr;
      }
    }
    
    const Channels &ChannelsInput::get_channels () const {
      return impl_->channels;
    }
    
//    void ChannelsInput::set_active_mask (const ChannelsHolder::ActiveChannelsMask &amask) {
//      if (impl_->channels)
//        impl_->channels->set_active_mask(amask);
//    }
    
    /***
     *
     * @param command_queue
     * @param destination
     * @param channels
     * @param transform
     * @param wait_until_completed
     * @param library_path
     */
    
    namespace impl {
        struct ChannelsOutputImpl {
            Channels channels = nullptr;
            //ChannelsDesc::ActiveChannelsMask amask;
            ChannelsDesc::Transform transform;
            //bool has_mask{};
            //Texture mask = nullptr;
        };
    }
    
    ChannelsOutput::ChannelsOutput(const void *command_queue,
                                   const Texture& destination,
                                   const Channels& channels,
                                   const ChannelsDesc::Transform& transform,
                                   //const ChannelsDesc::ActiveChannelsMask& amask,
                                   bool wait_until_completed,
                                   const std::string& library_path):
            Kernel(command_queue,
                   "one_channel_to_image",
                   nullptr,
                   destination,
                   wait_until_completed,
                   library_path),
            impl_(std::make_shared<impl::ChannelsOutputImpl>())
    {
      //impl_->amask = channels->;
      impl_->channels = channels;
      impl_->transform = transform;
      //impl_->has_mask = impl_->transform.mask != nullptr;
      
//      if (!impl_->transform.mask) {
//        TextureDesc desc ={
//                .width = 1,
//                .height = 1
//        };
//        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
//        impl_->mask = desc.make(get_command_queue(),mem);
//      }
    }
    
    void ChannelsOutput::process () {
      
      auto *channels = dynamic_cast<impl::ChannelsHolder *>(impl_->channels.get());
  
      if (get_source()){
        PassKernel(get_command_queue(),get_source(),get_destination(), get_wait_completed(), get_library_path()).process();
      }
      
      for (int j = 0; j < channels->size(); ++j) {
        
        auto channel = channels->at(j);
        
        //bool has_channel = channel != nullptr;
        
        if (!channel) continue;
  
        //channel = MemoryHolder::Make(get_command_queue(),1);
        
        execute([this, channels, &channel, j](CommandEncoder& encoder){
          
          auto src = get_source() ? get_source() : get_destination();
          
            encoder.set(src,0);
            
            encoder.set(get_destination(),1);
            
            encoder.set(channel,2);
            
            int cw = static_cast<int>(channels->get_width(j));
            int ch = static_cast<int>(channels->get_height(j));
            
            encoder.set(cw, 3);
            encoder.set(ch, 4);
            
            encoder.set(j, 5);
            
            encoder.set(impl_->transform.slope[j],6);
            encoder.set(impl_->transform.offset[j],7);
            
            if (impl_->transform.flags.out_enabled)
              encoder.set(impl_->transform.enabled[j],8);
            else
              encoder.set(false,8);
            
            encoder.set(impl_->transform.direction ,9);
            
            encoder.set(impl_->transform.type ,10);
            
//            encoder.set(impl_->has_mask , 11);
//            encoder.set(impl_->mask , 12);
//            encoder.set(has_channel , 13);
            
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
      impl_->transform = transform;
//      impl_->has_mask = transform.mask != nullptr;
//      if (!impl_->transform.mask) {
//        TextureDesc desc ={
//                .width = 1,
//                .height = 1
//        };
//        float mem[4] = {1.0f,1.0f,1.0f,1.0f};
//        impl_->mask = desc.make(get_command_queue(),mem);
//      }
//      else {
//        impl_->mask = impl_->transform.mask;
//      }
    }
    
    const ChannelsDesc::Transform &ChannelsOutput::get_transform () const {
      return impl_->transform;
    }
    
    void ChannelsOutput::set_channels (const Channels &channels) {
      impl_->channels = channels;
      //impl_->channels->set_active_mask(channels->get_active_mask());
    }
    
    const Channels &ChannelsOutput::get_channels () const {
      return impl_->channels;
    }
    
//    void ChannelsOutput::set_active_mask (const ChannelsHolder::ActiveChannelsMask &amask) {
//      if (impl_->channels)
//        impl_->channels->set_active_mask(amask);
//    }
}