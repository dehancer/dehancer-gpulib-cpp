//
// Created by denn nevera on 09/11/2020.
//

#include "dehancer/opencl/buffer.h"
#include "dehancer/opencl/embeddedProgram.h"
#include "gtest/gtest.h"

#include "dehancer/Texture.h"
#include "dehancer/Function.h"
#include "Image.h"

#include <algorithm>
#include <chrono>

cl_command_queue make_command_queue(const std::shared_ptr<clHelper::Device>& device) {
  /* Create OpenCL context */
  cl_int ret;

  cl_device_id device_id = device->clDeviceID;
  cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

  /* Create Command Queue */
  return clCreateCommandQueue(context, device_id, 0, &ret);
}


int run_bench2(int num, const std::shared_ptr<clHelper::Device>& device) {
  cl_uint width = 800, height = 600;

  auto command_queue = make_command_queue(device);

  auto bench_kernel = dehancer::Function(command_queue, "ao_bench_kernel");
  auto ao_bench_text = bench_kernel.make_texture(width,height);

  bench_kernel.execute([&ao_bench_text](dehancer::CommandEncoder& command_encoder){
      int numSubSamples = 4, count = 0;

      command_encoder.set(&numSubSamples, sizeof(numSubSamples), count++);
      command_encoder.set(ao_bench_text, count++);

      return ao_bench_text;
  });

  Image image(width, height);

  std::chrono::time_point<std::chrono::system_clock> clock_begin
          = std::chrono::system_clock::now();

  size_t originst[3];
  size_t regionst[3];
  size_t  rowPitch = 0;
  size_t  slicePitch = 0;
  originst[0] = 0; originst[1] = 0; originst[2] = 0;
  regionst[0] = width; regionst[1] = height; regionst[2] = 1;

  cl_int ret = clEnqueueReadImage(
          command_queue,
          static_cast<cl_mem>(ao_bench_text->get_contents()),
          CL_TRUE,
          originst,
          regionst,
          rowPitch,
          slicePitch,
          image.pix,
          0,
          nullptr,
          nullptr );

  if (ret != CL_SUCCESS) {
    std::runtime_error("Unable to create texture");
  }

  std::chrono::time_point<std::chrono::system_clock> clock_end
          = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = clock_end-clock_begin;

  // Report results and save image
  std::cout << "[aobench cl]:\t" << seconds.count() << "s "
            << ", for a " << width << "x" << height << " pixels" << std::endl;

  std::string out_file = "ao-cl-"; out_file.append(std::to_string(num)); out_file.append(".ppm");

  image.savePPM(out_file.c_str());

  return 0;
}


TEST(USER, OpenCL) {

  std::cout << std::endl;
  std::cerr << std::endl;

  try {
    std::vector<std::shared_ptr<clHelper::Device>> devices
            = clHelper::getAllDevices();

    std::shared_ptr<clHelper::Device> device;

    assert(!devices.empty());

    int dev_num = 0;
    std::cout << "Info: " << std::endl;
    for (auto d: devices) {
      std::cout << " #" << dev_num++ << std::endl;
      d->print(" ", std::cout);
    }

    std::cout << "Bench: " << std::endl;
    dev_num = 0;
    for (auto d: devices) {
      if (run_bench2(dev_num++, d)!=0) return;
    }

  }
  catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}