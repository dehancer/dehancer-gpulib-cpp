//
// Created by denn on 11.01.2021.
//

#include "dehancer/gpu/Filter.h"
#include "dehancer/gpu/operations/PassKernel.h"

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
            
            const void *command_queue{};
            Texture source = nullptr;
            Texture destination = nullptr;
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
            impl_(std::make_shared<impl::FilterImlp>()),
            name(),
            cache_enabled(false)
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
    
    Filter &Filter::add (const std::shared_ptr<Kernel> &kernel, bool enabled) {
      
      auto item = std::make_shared<impl::Item>((impl::Item){
              .kernel = kernel,
              .enabled = enabled
      });
      
      impl_->list.push_back(item);
      
      return *this;
    }
    
    Filter& Filter::process (bool emplace) {
      
      if (!impl_->source) return *this;
      
      auto current_source = impl_->source;
      
      TextureDesc desc = impl_->destination->get_desc();
      
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
          auto ping = TextureHolder::Make(impl_->command_queue, desc);
          impl_->ping_pong.at(0) = ping;
        }
      }
      
      int next_index = 0;
      
      auto pass_kernel = PassKernel(impl_->command_queue, impl_->wait_until_completed, impl_->library_path);
      
      int index = 0;
      bool make_last_copy = true;
      for (const auto& f: impl_->list) {
        
        auto current_destination = impl_->ping_pong[next_index%2]; next_index++;
  
        if (index==impl_->list.size()-1 && impl_->source->get_desc()==impl_->destination->get_desc()) {
          current_destination = impl_->destination;
          make_last_copy = false;
        } else {
          if (!current_destination || current_destination->get_desc() != desc) {
    
            desc.label = "Filter[" + get_name() + "] pong texture";
    
            auto pong = TextureHolder::Make(impl_->command_queue, desc);
            impl_->ping_pong.at(1) = pong;
            current_destination = pong;
          }
        }
        
        if (!f->enabled){
          pass_kernel.set_source(current_source);
          pass_kernel.set_destination(current_destination);
          pass_kernel.process();
        } else {
          if (f->kernel) {
            
            f->kernel->set_source(current_source);
            f->kernel->set_destination(current_destination);
            f->kernel->process();
          }
          
          else if (f->filter) {
            
            f->filter->set_source(current_source);
            f->filter->set_destination(current_destination);
            f->filter->process(f->emplace);
          }
        }
        
        current_source = current_destination;
        index++;
      }
      
      if (!emplace && impl_->destination && make_last_copy) {
        pass_kernel.set_source(current_source);
        pass_kernel.set_destination(impl_->destination);
        pass_kernel.process();
      }
  
      if (!cache_enabled)
        impl_->ping_pong = {nullptr, nullptr};
      
      return *this;
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
      if (index>=0 && index<impl_->list.size() )
        return impl_->list[index]->kernel;
      return Item(Error(CommonError::OUT_OF_RANGE));
    }
    
    bool Filter::is_enable (int index) const {
      if (index>=0 && index<impl_->list.size() )
        return impl_->list[index]->enabled;
      return false;
    }
    
    bool Filter::set_enable (int index, bool enabled) {
      if (index>=0 && index<impl_->list.size() ) {
        impl_->list[index]->enabled = enabled;
        return true;
      }
      return false;
    }
    
    bool Filter::is_enable (const Filter::FilterItem &item) const {
      int index = get_index_of(item);
      if(index>=0)
        return is_enable(index);
      return false;
    }
    
    bool Filter::is_enable (const Filter::KernelItem &item) const {
      int index = get_index_of(item);
      if(index>=0)
        return is_enable(index);
      return false;
    }
    
    bool Filter::set_enable (const Filter::FilterItem &item, bool enabled) {
      int index = get_index_of(item);
      if(index>=0)
        return set_enable(index, enabled);
      return false;
    }
    
    bool Filter::set_enable (const Filter::KernelItem &item, bool enabled) {
      int index = get_index_of(item);
      if(index>=0)
        return set_enable(index, enabled);
      return false;
    }
    
    int Filter::get_index_of (const Filter::FilterItem &item) const {
      for (int i = 0; i < impl_->list.size(); ++i) {
        if(auto f = impl_->list.at(i)->filter) {
          if (f.get() == item.get())
            return i;
        }
      }
      return -1;
    }
    
    int Filter::get_index_of (const Filter::KernelItem &item) const {
      for (int i = 0; i < impl_->list.size(); ++i) {
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
    
}