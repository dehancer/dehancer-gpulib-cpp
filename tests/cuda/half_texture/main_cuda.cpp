//
// Created by denn nevera on 15/11/2020.
//

#include "gtest/gtest.h"
//#include "dehancer/gpu/kernels/cuda/texture2d.h"
#include "dehancer/gpu/Lib.h"
#include "tests/cuda/paths_config.h"
#include "tests/shaders/test_struct.h"

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(static_cast<cudaError_t>(result)), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void print_device_info (CUdevice device) {
  
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, device);
  
  std::cout << "CUDA device id: " << device<< std::endl;
  std::cout << "CUDA device uid: " ;
  for(char byte : props.uuid.bytes)
    std::cout << std::hex << (int)byte;
  std::cout << std::endl;
  std::cout << "CUDA                device name: " << props.name << std::endl;
  std::cout << "CUDA device multiProcessorCount: " << (size_t)props.multiProcessorCount << std::endl;
  std::cout << "CUDA  device maxThreadsPerBlock: " << props.maxThreadsPerBlock << std::endl;
  std::cout << "CUDA       device maxThreadsDim: " << props.maxThreadsDim[0] << "x" << props.maxThreadsDim[1] << "x" << props.maxThreadsDim[2] << std::endl;
}


TEST(CUDA, Half_Texture) {
  
  try {
    int device_id = 0;
  
    print_device_info(device_id);
  
    auto command_queue = dehancer::DeviceCache::Instance().get_default_command_queue();//get_command_queue(dehancer::device::get_id(&device_id));
  
    int m = 4; // height = #rows
    int n = 3; // width  = #columns
  
    auto desc = (dehancer::TextureDesc) {
            .width = static_cast<size_t>(n),
            .height = static_cast<size_t>(m),
            .pixel_format = dehancer::TextureDesc::PixelFormat::rgba16float
    };
  
    auto texture = dehancer::TextureHolder::Make(command_queue, desc);
  
    auto function = dehancer::Function(command_queue, "kernel_half_test");
  
    function.execute([n, m, &texture] (dehancer::CommandEncoder &command_encoder) {
        command_encoder.set(texture,0);
        command_encoder.set(n, 1);
        command_encoder.set(m, 2);
        return dehancer::CommandEncoder::Size::From(1, 1);
    });
  
  
    dehancer::DeviceCache::Instance().return_command_queue(command_queue);
  }
  catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

namespace dehancer::device {
    
    /**
      * MUST BE defined in certain plugin module
      * @return metal lib path.
      */
    std::string get_lib_path() {
      return CUDA_KERNELS_LIBRARY;// + std::string("++");
    }
    
    extern std::size_t get_lib_source(std::string& source) {
      source.clear();
      return std::hash<std::string>{}(source);
    }
}

void stupid_draft_code() {
  
  std::cout << std::endl;
  std::cerr << std::endl;
  
  cuInit(0);
  
  int numDevices = 0;
  int maxMultiprocessors = 0, maxDevice = 0;
  
  cudaGetDeviceCount(&numDevices);
  
  std::cout << "CUDA device number: " << numDevices << std::endl;
  
  
  if (numDevices >= 1) {
    for (int device=0; device<numDevices; device++) {
      cudaDeviceProp props{};
      cudaGetDeviceProperties(&props, device);
      
      print_device_info(device);
      
      if (maxMultiprocessors < props.multiProcessorCount) {
        maxMultiprocessors = props.multiProcessorCount;
        maxDevice = device;
      }
    }
    cudaSetDevice(maxDevice);
  }
  
  cudaSetDevice(0);
  
  // Get handle for device 0
  CUdevice cuDevice;
  checkCudaErrors(cuDeviceGet(&cuDevice, 0));
  
  //cudaSetDevice(maxDevice);
  
  // Create context
  CUcontext cuContext;
  (cuCtxCreate(&cuContext, 0, cuDevice));
  
  cudaStream_t stream_0;
  checkCudaErrors(cudaStreamCreate(&stream_0));
  
  // Check stream
  CUcontext cuContext_0;
  checkCudaErrors(cuStreamGetCtx(stream_0, &cuContext_0));
  
  checkCudaErrors(cuCtxPopCurrent(&cuContext));
  
  checkCudaErrors(cuCtxPushCurrent(cuContext_0));
  
  CUdevice cUdevice_0 = -1;
  checkCudaErrors(cuCtxGetDevice(&cUdevice_0));
  
  cudaDeviceProp props{};
  
  cudaGetDeviceProperties(&props, cUdevice_0);
  
  std::cout << ">> checking ...." << std::endl;
  print_device_info(cUdevice_0);
  
  
  // Create module from binary file
  CUmodule cuModule;
  std::cout << "Module: " << dehancer::device::get_lib_path() << std::endl;
  checkCudaErrors(cuModuleLoad(&cuModule, dehancer::device::get_lib_path().c_str()));
  
  // Get function handle from module
  CUfunction half_texture_kernel;
  checkCudaErrors(cuModuleGetFunction(&half_texture_kernel, cuModule, "kernel_half_test"));
  
  int m = 4; // height = #rows
  int n = 3; // width  = #columns
  size_t pitch, tex_ofs;
  unsigned short arr[4][3]= {{0x0000,0x0001,0x0002},  // zero, denormals
                             {0x3c00,0x3c01,0x3c02},  // 1.0 + eps
                             {0x4000,0x4001,0x4002},  // 2.0 + eps
                             {0x7c00,0x7c01,0x7c02}}; // infinity, NaNs
  cudaArray *arr_d = nullptr;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDescHalf();
  
  checkCudaErrors(
          cudaMallocPitch(
                  (void**)&arr_d,
                  &pitch,
                  n*sizeof(unsigned short),
                  m)
  );
  
  checkCudaErrors(
          cudaMemcpy2D(
                  arr_d,
                  pitch,
                  arr,
                  n*sizeof(arr[0][0]),
                  n*sizeof(arr[0][0]),
                  m,
                  cudaMemcpyHostToDevice)
  );
  
  //texture<float, 2> tex;
  cudaTextureObject_t tex;
  
//  cudaResourceDesc resDesc{};
//  memset(&resDesc, 0, sizeof(resDesc));
//  resDesc.resType = cudaResourceTypePitch2D;
//  resDesc.res.pitch2D.devPtr = arr_d;
//  resDesc.res.pitch2D.desc = channelDesc;
//  resDesc.res.pitch2D.width = n;
//  resDesc.res.pitch2D.height = m;
//  resDesc.res.pitch2D.pitchInBytes = n*sizeof(unsigned short);
//
//  // Specify texture object parameters
//  cudaTextureDesc texDesc{};
//  memset(&texDesc, 0, sizeof(texDesc));
//  texDesc.addressMode[0]   = cudaAddressModeBorder;
//  texDesc.addressMode[1]   = cudaAddressModeBorder;
//  texDesc.filterMode       = cudaFilterModePoint;
//  texDesc.readMode         = cudaReadModeNormalizedFloat;
//  texDesc.normalizedCoords = true;
  
  cudaResourceDesc resDesc{};
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = arr_d;
  
  //--- Specify surface ---
  //CHECK_CUDA(cudaCreateSurfaceObject(&surface_, &resDesc));
  
  // Specify texture object parameters
  cudaTextureDesc texDesc{};
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeMirror;
  texDesc.addressMode[1]   = cudaAddressModeMirror;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = false;
  
  // Create texture object
  checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
  
//  checkCudaErrors (cudaBindTexture2D (&tex_ofs, &tex, arr_d, &channelDesc,
//                                     n, m, pitch));
  auto desc = (dehancer::TextureDesc){
    .width = static_cast<size_t>(n),
    .height = static_cast<size_t>(m)
  };
  auto tex1 = dehancer::TextureHolder::Make((void*)stream_0, desc);//std::make_shared<dehancer::nvcc::texture2d<float4,false>>(n,m);

  void* args[] = {tex1.get(), &n,&m};
  
  checkCudaErrors(cuLaunchKernel(
          half_texture_kernel,
          n, m, 1,
          1, 1, 1,
          0,
          stream_0,
          args,
          nullptr)
  );
  
  checkCudaErrors(cuCtxDestroy(cuContext));
  
}
