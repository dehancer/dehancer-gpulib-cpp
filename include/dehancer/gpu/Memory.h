//
// Created by denn nevera on 18/11/2020.
//

#pragma once

#include <memory>
#include "dehancer/Common.h"

namespace dehancer {
    
    struct MemoryHolder;
    
    /***
     * Memory pointer object
     */
    using Memory = std::shared_ptr<MemoryHolder>;
    
    struct MemoryDesc {
        enum MemType : uint32_t {
            host,
            device
        };
        
        size_t length{};
        MemType type = MemType::host;
        Memory make(const void *command_queue, const void* from_memory = nullptr);
    };
    
    /***
     * Device memory object holder
     */
    struct MemoryHolder : public std::enable_shared_from_this<MemoryHolder> {
        /***
         * Allocate new MemoryHolder object on device
         * @param command_queue - device command_queue or context
         * @param buffer - host memory buffer
         * @param length - buffer length in bytes
         * @return device memory object holder
         */
        static Memory Make(const void *command_queue, const void *buffer, size_t length);
        
        /***
        * Allocate new MemoryHolder object on device
        * @param command_queue - device command_queue or context
        * @param length - buffer length in bytes
        * @return device memory object holder
        */
        static Memory Make(const void *command_queue, size_t length);
        
        /***
        * Allocate new MemoryHolder object on device
        * @param command_queue - device command_queue or context
        * @param buffer - host memory bytes buffer
        * @return device memory object holder
        */
        static Memory Make(const void *command_queue, std::vector<uint8_t> buffer);
        
        /***
         * Create MemoryHolder object from device allocated object
         * @param command_queue - device command_queue or context
         * @param device_memory - device memory object handler
         * @return device memory object holder
         */
        static Memory Make(const void* command_queue, void* device_memory);
        
        static Memory Make(const void* command_queue, const void* device_memory);
        
        /***
        * Get a weak shared pointer to memory object.
        * @return
        */
        Memory get_ptr() { return shared_from_this(); }
        
        
        /***
         * Get memory object size in bytes.
         * @return number of bytes
         */
        virtual size_t get_length() const = 0;
        
        /***
        * Get platform specific handler of object placed in device memory.
        * @return device memory handler
        */
        [[nodiscard]] virtual const void*  get_memory() const = 0;
        
        /***
        * Get platform specific handler of object placed in device memory.
        * @return device memory handler
        */
        [[nodiscard]] virtual void*  get_memory() = 0;
        
        
        /***
         * Get memory object pointer
         * @return device memory object pointer
         */
        [[nodiscard]] virtual const void*  get_pointer() const = 0;
        [[nodiscard]] virtual void*  get_pointer() = 0;
        
        /***
         * Copy contents of memory object to host memory buffer as as a sequential array of bytes.
         * @param buffer
         * @return expected Error object descriptor or Error::OK
         */
        virtual Error get_contents(std::vector<uint8_t>& buffer) const = 0;
        virtual Error get_contents(void *buffer, size_t length) const = 0;
    
        template<class T>
        Error get_contents(std::vector<T>& buffer) {
          if (get_length()==0) {
            buffer.clear();
            return Error(CommonError::OK);
          }
          size_t length = get_length();
          size_t size = length / sizeof(T);
          buffer.resize(size);
          return get_contents(buffer.data(), length);
        };
    
        virtual ~MemoryHolder() = default;
    
    protected:
        MemoryHolder() = default;
    };
    
}