//
// Created by denn on 11.01.2021.
//

#include "dehancer/gpu/Filter.h"
#include "dehancer/gpu/operations/PassKernel.h"
#include "dehancer/gpu/Log.h"

namespace dehancer {
    
    namespace impl {
        
        template <typename T>
        std::string filter_name(T t)
        {
          return typeid(t).name();
        }
        
        struct Item {
            
            std::shared_ptr<dehancer::Kernel> kernel = nullptr;
            std::shared_ptr<dehancer::Filter> filter = nullptr;
            bool    enabled = true;
            bool    emplace = false;
        };
        
        struct FilterImlp {
            
            const void *command_queue = nullptr;
            Texture     source = nullptr;
            Texture     destination = nullptr;
            bool        wait_until_completed = false;
            std::string library_path;
            
            std::vector<std::shared_ptr<Item>> list{};
            
            std::array<Texture,2> ping_pong = {nullptr, nullptr};
        };
    }
    
    Filter::Filter (const void *command_queue,
                    const Texture &source,
                    const Texture &destination,
                    bool wait_until_completed,
                    const std::string &library_path
    ):
            name(),
            cache_enabled(false),
            impl_(std::make_shared<impl::FilterImlp>())
    {
      impl_->command_queue = command_queue;
      impl_->source = source;
      impl_->destination = destination;
      impl_->wait_until_completed = wait_until_completed;
      impl_->library_path = library_path;
    }
    
    Filter &Filter::add (const Filter::FilterItem &filter, bool enabled, bool emplace) {
      auto item = std::make_shared<impl::Item>((impl::Item){
              .filter = filter,
              .enabled = enabled,
              .emplace = emplace
      });
      impl_->list.push_back(item);
      return *this;
    }
    
    Filter &Filter::add (const std::shared_ptr<Kernel> &kernel, bool enabled, bool emplace) {
      
      auto item = std::make_shared<impl::Item>((impl::Item){
              .kernel = kernel,
              .enabled = enabled,
              .emplace = emplace
      });
      
      impl_->list.push_back(item);
      
      return *this;
    }
    
    
    Filter& Filter::process (bool emplace) {
      
      if (!impl_->source) return *this;
      
      if (impl_->list.empty()) {
        
        if (impl_->source == impl_->destination) {
          return *this;
        }
        PassKernel(impl_->command_queue,
                   impl_->source,
                   impl_->destination,
                   impl_->wait_until_completed,
                   impl_->library_path).process();
        return *this;
      }
      
      auto current_source = impl_->source;
      
      auto desc = impl_->destination->get_desc();
      
      if (emplace) {
        if (impl_->destination)
          impl_->ping_pong = {impl_->destination, impl_->destination};
        else
          impl_->ping_pong = {impl_->source, impl_->source};
      }
      else {
        
        if (current_source->get_length() < impl_->destination->get_length())
          desc = current_source->get_desc();
        
        if (!impl_->ping_pong.at(0) || impl_->ping_pong.at(0)->get_desc()!=desc) {
          desc.label = "Filter[" + get_name() + "] ping texture";
          desc.mem_flags = TextureDesc::MemFlags::less_memory;
          auto ping = TextureHolder::Make(impl_->command_queue, desc);
          impl_->ping_pong.at(0) = ping;
        }
      }
      
      int next_index = 0;
      
      int index = 0;
      
      bool make_last_copy = true;

#pragma unroll
      for (const auto& f: impl_->list) {
        
        if (!f->enabled) continue;
        
        auto current_destination = impl_->ping_pong[next_index%2]; next_index++;
        
        if (index==(int)impl_->list.size()-1 && impl_->source->get_desc()==impl_->destination->get_desc()) {
          current_destination = impl_->destination;
          make_last_copy = false;
        } else {
          if (!current_destination || current_destination->get_desc() != desc) {
            
            desc.label = "Filter[" + get_name() + "] pong texture";
            desc.mem_flags = TextureDesc::MemFlags::less_memory;
            
            auto pong = TextureHolder::Make(impl_->command_queue, desc);
            impl_->ping_pong.at(1) = pong;
            current_destination = pong;
          }
        }
        
        if (f->kernel) {
          f->kernel->process(current_source, current_destination);
        }
        
        else if (f->filter) {
          f->filter->process(current_source, current_destination, f->emplace);
        }
        
        current_source = current_destination;
        index++;
      }
      
      if ((!emplace || index==0) && impl_->destination && make_last_copy) {
        PassKernel(impl_->command_queue,
                   current_source,
                   impl_->destination,
                   impl_->wait_until_completed,
                   impl_->library_path).process();
      }
      
      if (!cache_enabled) {
        impl_->ping_pong = {nullptr, nullptr};
      }
      
      return *this;
    }
    
    Filter &Filter::process(const Texture& source, const Texture& destination, bool emplace) {
      set_source(source);
      set_destination(destination);
      return process(emplace);
    }
    
    Filter &Filter::process (const Texture &source, const Texture &destination) {
      return process(source,destination, false);
    }
    
    const Texture &Filter::get_source () const {
      return impl_->source;
    }
    
    const Texture &Filter::get_destination () const {
      return impl_->destination;
    }
    
    
    void Filter::set_source (const Texture &source) {
      impl_->source = source;
    }
    
    void Filter::set_destination (const Texture &destination) {
      impl_->destination = destination;
    }
    
    Filter::Item Filter::get_item_at (int index) const {
      if (index>=0 && index<(int)impl_->list.size() )
        return impl_->list[index]->kernel;
      return Item(Error(CommonError::OUT_OF_RANGE));
    }
    
    bool Filter::is_enable (int index) const {
      if (index>=0 && index<(int)impl_->list.size() )
        return impl_->list[index]->enabled;
      return false;
    }
    
    bool Filter::set_enable (int index, bool enabled) {
      if (index>=0 && index<(int)impl_->list.size() ) {
        impl_->list[index]->enabled = enabled;
        return true;
      }
      return false;
    }
    
    bool Filter::is_enable (const Filter::FilterItem &item) const {
      int const index = get_index_of(item);
      if(index>=0)
        return is_enable(index);
      return false;
    }
    
    bool Filter::is_enable (const Filter::KernelItem &item) const {
      int const index = get_index_of(item);
      if(index>=0)
        return is_enable(index);
      return false;
    }
    
    bool Filter::set_enable (const Filter::FilterItem &item, bool enabled) {
      int const index = get_index_of(item);
      if(index>=0)
        return set_enable(index, enabled);
      return false;
    }
    
    bool Filter::set_enable (const Filter::KernelItem &item, bool enabled) {
      int const index = get_index_of(item);
      if(index>=0)
        return set_enable(index, enabled);
      return false;
    }
    
    int Filter::get_index_of (const Filter::FilterItem &item) const {
#pragma unroll
      for (int i = 0; i < (int)impl_->list.size(); ++i) {
        if(auto f = impl_->list.at(i)->filter) {
          if (f.get() == item.get())
            return i;
        }
      }
      return -1;
    }
    
    int Filter::get_index_of (const Filter::KernelItem &item) const {
#pragma unroll
      for (int i = 0; i < (int)impl_->list.size(); ++i) {
        if(auto k = impl_->list.at(i)->kernel) {
          if (k.get() == item.get())
            return i;
        }
      }
      return -1;
    }
    
    std::string Filter::get_name () const {
      if (name.empty())
        return impl::filter_name(*this);
      return name;
    }
    
    const void *Filter::get_command_queue () const {
      return impl_->command_queue;
    }
}