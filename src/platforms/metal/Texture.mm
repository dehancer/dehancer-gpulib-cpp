//
// Created by denn nevera on 10/11/2020.
//

#include "Texture.h"
#include <cstring>

namespace dehancer::metal {

    TextureHolder::TextureHolder(const void *command_queue, const TextureDesc &desc, void *from_memory) :
            Context(command_queue),
            desc_(desc),
            texture_(nullptr)
    {

      MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor new] autorelease];

      descriptor.width  = (NSUInteger)desc.width;
      descriptor.height = (NSUInteger)desc.height;
      descriptor.depth  = (NSUInteger)desc.depth;
      descriptor.arrayLength = 1;
      descriptor.mipmapLevelCount = 1;
      descriptor.storageMode = MTLStorageModeManaged;

      descriptor.usage = MTLTextureUsagePixelFormatView|MTLTextureUsageRenderTarget;
      descriptor.usage |= desc.mem_flags&TextureDesc::MemFlags::read_only ? MTLTextureUsageShaderRead : 0;
      descriptor.usage |= desc.mem_flags&TextureDesc::MemFlags::write_only ? MTLTextureUsageShaderRead : 0;
      descriptor.usage |= desc.mem_flags&TextureDesc::MemFlags::read_write ? MTLTextureUsageShaderWrite|MTLTextureUsageShaderRead : 0;
      descriptor.storageMode = MTLStorageModeManaged;

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
          descriptor.pixelFormat = MTLPixelFormatRGBA8Uint;
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
        buffer = reinterpret_cast<unsigned char *>(from_memory);
      }

      texture_ = [get_command_queue().device newTextureWithDescriptor:descriptor];

      if (!texture_)
        throw std::runtime_error("Unable to create texture");

      if (buffer) {

        NSUInteger bytes_per_pixel = desc.channels * componentBytes;

        [texture_ replaceRegion: region
                    mipmapLevel: 0
                          slice: 0
                      withBytes: buffer
                    bytesPerRow: bytes_per_pixel * region.size.width
                  bytesPerImage: bytes_per_pixel * region.size.width * region.size.height];
      }
    }

    dehancer::Error TextureHolder::get_contents(std::vector<float>& buffer) const {

      auto componentBytes = sizeof(Float32);

      switch (desc_.pixel_format) {

        case TextureDesc::PixelFormat::rgba32float:
          componentBytes = sizeof(Float32);
          break;

        default:
          return Error(CommonError::NOT_SUPPORTED, "Texture should be rgba32float");
      }

      id<MTLCommandQueue> queue = get_command_queue();

      id <MTLCommandBuffer> commandBuffer = [queue commandBuffer];

      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      [blitEncoder synchronizeTexture:texture_ slice:0 level:0];
      [blitEncoder endEncoding];

      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

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

      buffer.resize(desc_.width*desc_.depth*desc_.height*desc_.channels);

      [texture_ getBytes: buffer.data()
             bytesPerRow: bytes_per_pixel * region.size.width
              fromRegion: region
             mipmapLevel: 0];

      return Error(CommonError::OK);

    }

    const void* TextureHolder::get_memory() const {
      return texture_;
    }

    void *TextureHolder::get_memory() {
      return texture_;
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

    TextureHolder::~TextureHolder() {
      if (texture_)
        [texture_ release];
    }
}