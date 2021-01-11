//
// Created by denn on 11.01.2021.
//

#include "dehancer/gpu/Filter.h"
#include "dehancer/gpu/operations/PassKernel.h"

namespace dehancer {
    
    namespace impl {
        
        struct KernelItem {
            
            std::shared_ptr<dehancer::Kernel> kernel = nullptr;
            bool    enabled = true;
        };
        
        struct FilterImlp {
            
            const void *command_queue{};
            Texture source = nullptr;
            Texture destination = nullptr;
            bool        wait_until_completed = false;
            std::string library_path;
            
            std::vector<std::shared_ptr<KernelItem>> list{};
            
            std::array<Texture,2> ping_pong = {nullptr, nullptr};
        };
    }
    
    Filter::Filter (const void *command_queue,
                    const Texture &source,
                    const Texture &destination,
                    bool wait_until_completed,
                    const std::string &library_path
    ):
            impl_(std::make_shared<impl::FilterImlp>())
    {
      impl_->command_queue = command_queue;
      impl_->source = source;
      impl_->destination = destination;
      impl_->wait_until_completed = wait_until_completed;
      impl_->library_path = library_path;
    }
    
    Filter &Filter::add (const std::shared_ptr<Kernel> &kernel, bool enabled) {
      
      auto item = std::make_shared<impl::KernelItem>((impl::KernelItem){
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
  
        auto ping = TextureHolder::Make(impl_->command_queue, desc);
  
        impl_->ping_pong.at(0) = ping;
      }
      
      int next_index = 0;
      
      auto pass_kernel = PassKernel(impl_->command_queue, impl_->wait_until_completed, impl_->library_path);
      
      for (const auto& f: impl_->list) {
        
        auto current_destination = impl_->ping_pong[next_index%2]; next_index++;
  
        if (!current_destination) {
          auto pong = TextureHolder::Make(impl_->command_queue, desc);
          impl_->ping_pong.at(1) = pong;
          current_destination = pong;
        }
        
        if (!f->enabled){
          pass_kernel.set_source(current_source);
          pass_kernel.set_destination(current_destination);
          pass_kernel.process();
        } else {
          std::cout << "Process Filter kernel: " << f->kernel->get_name() << " enabled: " << f->enabled << " emplace: "
                    << emplace << std::endl;
          f->kernel->set_source(current_source);
          f->kernel->set_destination(current_destination);
          f->kernel->process();
        }
        
        current_source = current_destination;
      }
  
      if (!emplace && impl_->destination) {
        pass_kernel.set_source(current_source);
        pass_kernel.set_destination(impl_->destination);
        pass_kernel.process();
      }
      
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
    
    std::shared_ptr<Kernel> Filter::get_kernel_at (int index) const {
      if (index>=0 && index<impl_->list.size() )
        return impl_->list[index]->kernel;
      return nullptr;
    }
    
    bool Filter::get_enabling_at (int index) const {
      if (index>=0 && index<impl_->list.size() )
        return impl_->list[index]->enabled;
      return false;
    }
    
    bool Filter::set_enabling_at (int index, bool enabled) {
      if (index>=0 && index<impl_->list.size() ) {
        impl_->list[index]->enabled = enabled;
        return true;
      }
      return false;
    }
  
}