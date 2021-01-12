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
    
        /***
         * Shared instance of gpu kernel
         */
        using KernelItem = std::shared_ptr<Kernel>;
        
        /***
         * Shared instance of filter
         */
        using FilterItem = std::shared_ptr<Filter>;
        
        /***
         * Filter item
         */
        using Item = std::variant<KernelItem,FilterItem,dehancer::Error>;
        
        /***
         * Create filter group that is a kernels or another filters conscience transforms
         * the source texture (rgba image) to destination texture.
         *
         * @param command_queue -  platform based queue handler
         * @param source - source kernel texture
         * @param destination - destination texture
         * @param wait_until_completed - flag defines completion state
         * @param library_path - explicit shaders library file path, resource name or source bundle
         *                      (opencl source can by name of embedded value)
         */
        explicit Filter(const void* command_queue,
                        const Texture& source= nullptr,
                        const Texture& destination= nullptr,
                        bool wait_until_completed = Command::WAIT_UNTIL_COMPLETED,
                        const std::string &library_path=""
        );
    
        virtual ~Filter() = default;
        
        /***
         * Add GPU Kernel instance
         * @param kernel - use std::make_shared<dehancer::Kernel> to create appropriate kernel object
         * @param enabled - initial state of processing, true means the kernel processes source, false do nothing
         * @return the current Filter object
         */
        Filter &add (const KernelItem &kernel, bool enabled = true);
        
        /***
         * Add GPU Filter instance
         * @param filter - use std::make_shared<dehancer::Kernel> to create appropriate filter object
         * @param enabled - initial state of processing, true means the kernel processes source, false do nothing
         * @param emplace - when is true destination is processed in-place their texture and no additional memory is allocated
         * @return the current Filter object
         */
        Filter &add (const FilterItem &filter, bool enabled = true, bool emplace = false);
    
        /***
         * Get Item object at indes
         * @param index - object item index
         * @return Item object or Error(CommonError::OUT_OF_RANGE)
         */
        Item get_item_at(int index) const;
        
        int get_index_of(const FilterItem& item) const;
        
        int get_index_of(const KernelItem& item) const;
    
        /***
         * Test enable state of kernel or filter placed at index of processing stack
         * @param index - object item index
         * @return state value, or index out of range
         */
        bool is_enable(int index) const;
    
        bool is_enable(const FilterItem& item) const;
        bool is_enable(const KernelItem& item) const;
    
        /***
         * Set enable state of kernel or filter placed at index of processing stack
         * @param index - object item index
         * @param enabled - state
         * @return true if state is changed, false if index out of range
         */
        bool set_enable(int index, bool enabled);
        bool set_enable(const FilterItem& item, bool enabled);
        bool set_enable(const KernelItem& item, bool enabled);
    
        /***
         * Process filter
         * @param emplace - when is true destination is processed in-place their texture and no additional memory is allocated
         * @return - the Filter object
         */
        virtual Filter& process(bool emplace);
        
        /***
         * Process filter
         * @return safe processing all source and destination keeps without changes on all processing stages
         * except last when result of processing writes in destination texture
         */
        virtual Filter& process() { return process(false); };
        
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

