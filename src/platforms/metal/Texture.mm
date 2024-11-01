//
// Created by denn nevera on 10/11/2020.
//

#include "Texture.h"
#include "dehancer/gpu/Log.h"
#import <Metal/Metal.h>

namespace dehancer::metal {
    
    TextureItem::~TextureItem(){
      if (texture && releasable) {
//        dehancer::log::print( "Metal make texture FREE: %p, %dx%d", texture, [static_cast<id <MTLTexture>>(texture) width],  [static_cast<id <MTLTexture>>(texture) height]);
        [static_cast<id <MTLTexture>>(texture) release];
      }
    }
    
    const void *TextureHolder::get_command_queue () const {
      return command_queue_;
    }
    
    TextureHolder::TextureHolder (const void *command_queue, const void *from_memory):
            dehancer::TextureHolder(),
            Context(command_queue),
            desc_(),
            texture_item_(nullptr),
            command_queue_((void*)command_queue)
    {
      
      if (!from_memory) return;
      
      texture_item_ = std::make_shared<TextureItem>();
      
      texture_item_->texture = static_cast<id <MTLTexture>>(from_memory);
      
      [static_cast<id <MTLTexture>>(texture_item_->texture) retain];
      
      switch ([static_cast<id <MTLTexture>>(texture_item_->texture) textureType]) {
        case MTLTextureType1D:
          desc_.type = TextureDesc::Type::i1d;
          break;
        
        case MTLTextureType2D:
          desc_.type = TextureDesc::Type::i2d;
          break;
        
        case MTLTextureType3D:
          desc_.type = TextureDesc::Type::i3d;
          break;
        
        default:
          throw std::runtime_error("Unsupported texture type");
      }
      
      switch ([static_cast<id <MTLTexture>>(texture_item_->texture) pixelFormat]) {
        case MTLPixelFormatRGBA32Float:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba32float;
          break;
        
        case MTLPixelFormatRGBA16Float:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba16float;
          break;
        
        case MTLPixelFormatRGBA32Uint:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba32uint;
          break;
        
        case MTLPixelFormatRGBA16Uint:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba16uint;
          break;
        
        case MTLPixelFormatRGBA8Uint:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
          desc_.pixel_format = TextureDesc::PixelFormat::rgba8uint;
          break;
        
        default:
          throw std::runtime_error("Unsupported texture pixel format");
      }
      
      desc_.width = [static_cast<id <MTLTexture>>(texture_item_->texture) width];
      desc_.height = [static_cast<id <MTLTexture>>(texture_item_->texture) height];
      desc_.depth = [static_cast<id <MTLTexture>>(texture_item_->texture) depth];
      desc_.channels = 4;
      
      texture_item_->hash = desc_.get_hash();
      
    }
    
    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc, const void *from_memory, bool is_device_buffer) :
            dehancer::TextureHolder(),
            Context(command_queue),
            desc_(desc),
            texture_item_(nullptr),
            command_queue_((void*)command_queue)
    {
      
      auto text_hash = desc_.get_hash();
      
      auto info = get_texture_info(desc.type);
      
      if (desc.width>info.max_width || desc.height>info.max_height || desc.depth>info.max_depth) {
        auto mess = message_string("GPU runtime error: image size limit. \r\n");
        dehancer::log::error(true, "Metal make texture limit error: %s", mess.c_str());
        throw dehancer::texture::memory_exception(mess);
      }
      
      MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor new] autorelease];
      
      descriptor.width  = (NSUInteger)desc.width;
      descriptor.height = (NSUInteger)desc.height;
      descriptor.depth  = (NSUInteger)desc.depth;
      descriptor.arrayLength = 1;
      descriptor.mipmapLevelCount = 1;
      
      descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
  
      if (desc.mem_flags&TextureDesc::MemFlags::less_memory) {
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      }
      else {
        
        if (has_unified_memory()){
          descriptor.storageMode = MTLStorageModeShared;
          descriptor.resourceOptions = MTLResourceCPUCacheModeDefaultCache;
        }
        else {
          #if defined(IOS_SYSTEM)
          descriptor.storageMode = MTLStorageModeShared;
          #else
          descriptor.storageMode = MTLStorageModeManaged;
          #endif
        }
        
        descriptor.usage |= desc.mem_flags & TextureDesc::MemFlags::read_only ? MTLTextureUsageShaderRead : 0;
        descriptor.usage |= desc.mem_flags & TextureDesc::MemFlags::write_only ? MTLTextureUsageShaderRead : 0;
        descriptor.usage |= desc.mem_flags & TextureDesc::MemFlags::read_write ? MTLTextureUsageShaderWrite |
                                                                                 MTLTextureUsageShaderRead : 0;
      }
      
      descriptor.allowGPUOptimizedContents = false;
      
      auto componentBytes = sizeof(Float32);
      
      MTLRegion region;
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          descriptor.pixelFormat = MTLPixelFormatRGBA32Float;
          componentBytes = sizeof(Float32);
          break;
        
        case TextureDesc::PixelFormat::rgba16float:
          descriptor.pixelFormat = MTLPixelFormatRGBA16Float;
          componentBytes = sizeof(Float32)/2;
          break;
        
        case TextureDesc::PixelFormat::rgba32uint:
          descriptor.pixelFormat = MTLPixelFormatRGBA32Uint;
          componentBytes = sizeof(uint32_t);
          break;
        
        case TextureDesc::PixelFormat::rgba16uint:
          descriptor.pixelFormat = MTLPixelFormatRGBA16Uint;
          componentBytes = sizeof(uint16_t);
          break;
        
        case TextureDesc::PixelFormat::rgba8uint:
          descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
          componentBytes = sizeof(uint8_t);
          break;
      }
      
      switch (desc_.type) {
        case TextureDesc::Type::i1d:
          descriptor.textureType = MTLTextureType1D;
          region = MTLRegionMake1D(0, desc_.width);
          break;
        case TextureDesc::Type::i2d:
          descriptor.textureType = MTLTextureType2D;
          region = MTLRegionMake2D(0, 0, desc_.width, desc_.height);
          break;
        case TextureDesc::Type::i3d:
          descriptor.textureType = MTLTextureType3D;
          descriptor.arrayLength = 1;
          descriptor.mipmapLevelCount = 1;
          region = MTLRegionMake3D(0, 0, 0, desc_.width, desc_.height, desc_.depth);
          break;
      }
      
      unsigned char* buffer = nullptr;
      
      if (from_memory) {
        buffer = reinterpret_cast<unsigned char *>((void *)from_memory);
      }
  
      auto device = static_cast<id<MTLDevice>>( (__bridge id) get_device());
      auto queue = static_cast<id<MTLCommandQueue>>( (__bridge id) get_command_queue());
      
      id <MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
      
      if (!texture)
        throw std::runtime_error("Unable to create texture");
      
      texture_item_ = std::make_shared<TextureItem>();
      
      texture_item_->hash = text_hash;
      texture_item_->texture = static_cast<id <MTLTexture>>(texture);
      
      if (buffer) {
        
        NSUInteger bytes_per_pixel = desc.channels * componentBytes;
  
        if (is_device_buffer){
          auto buff = reinterpret_cast<id<MTLBuffer> >((__bridge id)buffer);
          
          id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
          
          id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
          
          [blitEncoder copyFromBuffer: buff
                         sourceOffset: 0
                    sourceBytesPerRow: bytes_per_pixel * region.size.width
                  sourceBytesPerImage: bytes_per_pixel * region.size.width * region.size.height
                           sourceSize: (MTLSize){desc_.width, desc_.height, desc_.depth}
                            toTexture: static_cast<id <MTLTexture>>(texture_item_->texture)
                     destinationSlice: 0
                     destinationLevel: 0
                    destinationOrigin: (MTLOrigin){0,0,0}
          ];
          
          [blitEncoder endEncoding];
          
          [commandBuffer commit];
          
        }
        else {
          [static_cast<id <MTLTexture>>(texture_item_->texture) replaceRegion:region
                                    mipmapLevel:0
                                          slice:0
                                      withBytes:buffer
                                    bytesPerRow:bytes_per_pixel * region.size.width
                                  bytesPerImage:bytes_per_pixel * region.size.width * region.size.height];
        }
        
        id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
  
        #if not defined(IOS_SYSTEM)
          id <MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
          [blitEncoder synchronizeTexture:static_cast<id <MTLTexture>>(texture_item_->texture) slice:0 level:0];
          [blitEncoder endEncoding];
        #endif
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
      }
    }
    
    dehancer::Error TextureHolder::get_contents(std::vector<float>& buffer) const {
      buffer.resize( get_length()/sizeof(float) );
      return get_contents(buffer.data(), get_length());
    }
    
    dehancer::Error TextureHolder::get_contents(void *buffer, size_t length) const {
      
      auto componentBytes = sizeof(Float32);
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          componentBytes = sizeof(Float32);
          break;
  
        case TextureDesc::PixelFormat::rgba16float:
          componentBytes =sizeof(Float32)/2;
          break;
  
        default:
          return Error(CommonError::NOT_SUPPORTED, "Texture should be rgba16float, rgba32float");
      }
      
      if (length< this->get_length()) {
        return Error(CommonError::OUT_OF_RANGE, "Texture length greater then buffer length");
      }
      
      auto queue = static_cast<id<MTLCommandQueue>>( (__bridge id) get_command_queue());
      
      id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
  
      #if not defined(IOS_SYSTEM)
      #if defined(DEHANCER_ARCH_IS_X86_64)
        id <MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder synchronizeTexture:static_cast<id <MTLTexture>>(texture_item_->texture) slice:0 level:0];
        [blitEncoder endEncoding];
      #endif
      #endif
      
      [commandBuffer commit];
      #if defined(DEHANCER_ARCH_IS_X86_64)
        [commandBuffer waitUntilCompleted];
      #endif
      
      MTLRegion region;
      
      switch (desc_.type) {
        case TextureDesc::Type::i1d:
          region = MTLRegionMake1D(0, desc_.width);
          break;
        case TextureDesc::Type::i2d:
          region = MTLRegionMake2D(0, 0, desc_.width, desc_.height);
          break;
        case TextureDesc::Type::i3d:
          region = MTLRegionMake3D(0, 0, 0, desc_.width, desc_.height, desc_.depth);
          break;
      }
      
      NSUInteger bytes_per_pixel = desc_.channels * componentBytes;
      
      [static_cast<id <MTLTexture>>(texture_item_->texture) getBytes: buffer
                           bytesPerRow: bytes_per_pixel * region.size.width
                            fromRegion: region
                           mipmapLevel: 0];
      
      return Error(CommonError::OK);
      
    }
    
    const void* TextureHolder::get_memory() const {
      return texture_item_->texture;
    }
    
    void *TextureHolder::get_memory() {
      return texture_item_->texture;
    }
    
    size_t TextureHolder::get_width() const {
      return desc_.width;
    }
    
    size_t TextureHolder::get_height() const {
      return desc_.height;
    }
    
    size_t TextureHolder::get_depth() const {
      return desc_.depth;
    }
    
    size_t TextureHolder::get_channels() const {
      return desc_.channels;
    }
    
    size_t TextureHolder::get_length() const {
      
      size_t size = desc_.width*desc_.depth*desc_.height*desc_.channels;
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          return size * sizeof(float);
        
        case TextureDesc::PixelFormat::rgba16float:
          return size * sizeof(float)/2;
        
        case TextureDesc::PixelFormat::rgba32uint:
          return size * sizeof(uint32_t);
        
        case TextureDesc::PixelFormat::rgba16uint:
          return size * sizeof(uint16_t);
        
        case TextureDesc::PixelFormat::rgba8uint:
          return size * sizeof(uint8_t);
      }
    }
    
    TextureDesc::PixelFormat TextureHolder::get_pixel_format() const {
      return desc_.pixel_format;
    }
    
    TextureDesc::Type TextureHolder::get_type() const {
      return desc_.type;
    }
    
    TextureHolder::~TextureHolder() = default;
    
    dehancer::Error TextureHolder::copy_to_device (void *buffer) const {
    
      auto componentBytes = sizeof(Float32);
      
      switch (desc_.pixel_format) {
        
        case TextureDesc::PixelFormat::rgba32float:
          componentBytes = sizeof(Float32);
          break;
        
        default:
          return Error(CommonError::NOT_SUPPORTED, "Texture should be rgba32float");
      }
      
      if (!buffer) {
        return Error(CommonError::OUT_OF_RANGE, "Target buffer isundefined");
      }
  
  
      MTLRegion region;
  
      switch (desc_.type) {
        case TextureDesc::Type::i1d:
          region = MTLRegionMake1D(0, desc_.width);
          break;
        case TextureDesc::Type::i2d:
          region = MTLRegionMake2D(0, 0, desc_.width, desc_.height);
          break;
        case TextureDesc::Type::i3d:
          region = MTLRegionMake3D(0, 0, 0, desc_.width, desc_.height, desc_.depth);
          break;
      }
  
      auto buff = reinterpret_cast<id<MTLBuffer> >((__bridge id)buffer);
  
      NSUInteger bytes_per_pixel = desc_.channels * componentBytes;
  
      auto queue = static_cast<id<MTLCommandQueue>>( (__bridge id) get_command_queue());
      
      id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];
      
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
  
  
      [blitEncoder copyFromTexture: static_cast<id <MTLTexture>>(texture_item_->texture)
                       sourceSlice: 0
                       sourceLevel: 0
                      sourceOrigin: (MTLOrigin) {0,0,0}
                        sourceSize: (MTLSize) {desc_.width,desc_.height,desc_.depth}
                          toBuffer: buff
                 destinationOffset: 0
            destinationBytesPerRow:(NSUInteger)bytes_per_pixel * region.size.width
          destinationBytesPerImage:(NSUInteger)bytes_per_pixel * region.size.width * region.size.height];
      
      [blitEncoder endEncoding];
      
      [commandBuffer commit];
      
      return Error(CommonError::OK);
    }
  
  
}