//
// Created by denn on 11.01.2021.
//

#pragma once

#include "dehancer/gpu/Kernel.h"
#include <variant>

namespace dehancer {
    
    namespace impl {
        struct FilterImlp;
    }
    
    /***
     * Filter makes a group of GPU compute functions.
     */
    class Filter: public std::enable_shared_from_this<Filter>  {
    public:
    
        using KernelItem = std::shared_ptr<Kernel>;
        using FilterItem = std::shared_ptr<Filter>;
        using Item = std::variant<KernelItem,FilterItem>;
        
        explicit Filter(const void* command_queue,
                        const Texture& source= nullptr,
                        const Texture& destination= nullptr,
                        bool wait_until_completed = Command::WAIT_UNTIL_COMPLETED,
                        const std::string &library_path=""
        );
    
        virtual ~Filter() = default;
        
        Filter &add (const KernelItem &kernel, bool enabled = true);
        Filter &add (const FilterItem &filter, bool enabled = true, bool emplace = false);
    
        Item get_item_at(int index) const;
        
        bool get_enabling_at(int index) const;
        
        bool set_enabling_at(int index, bool enabled);
        
        virtual Filter& process(bool emplace);
        Filter& process() { return process(false); };
        
        /**
         * Get source texture
         * @return texture object
         */
        [[nodiscard]] virtual const Texture& get_source() const;
        
        /***
         * Get destination texture
         * @return texture object
         */
        [[nodiscard]] virtual const Texture& get_destination() const;
        
        /***
         * Set new source
         *
         */
        virtual void set_source(const Texture& source);
        
        /***
         * Set new destination texture
         * @param dest - texture object
         */
        virtual void set_destination(const Texture& destination);
        
        std::shared_ptr<Filter> get_ptr() { return shared_from_this(); }
    
    private:
        std::shared_ptr<impl::FilterImlp> impl_;
    };
}

