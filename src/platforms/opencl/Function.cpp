//
// Created by denn nevera on 10/11/2020.
//

#include "Function.h"
#include "CommandEncoder.h"
#include "dehancer/gpu/Paths.h"

namespace dehancer::opencl {

    std::mutex Function::mutex_;
    std::unordered_map<cl_command_queue, Function::KernelMap> Function::kernel_map_;
    std::unordered_map<cl_command_queue, Function::ProgamMap> Function::program_map_;

    void Function::execute(const dehancer::Function::FunctionHandler &block) {

      std::unique_lock<std::mutex> lock(Function::mutex_);

      auto texture_size = block(*encoder_);
      
      //size_t local_work_size[3] = {16,16,16};
      size_t local_work_size[3] = {1,1,1};
      
      ///
      /// TODO: optimize workgroups automatically
      ///
      //clGetKernelWorkGroupInfo(kernel_, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), local_work_size, nullptr);
      //local_work_size[1] = 1;
      //if (local_work_size[0]>=texture_size.width) local_work_size[0] = 1;

//      if (texture_size.depth==1) {
//        local_work_size[2] = 1;
//      }
//      else {
//        local_work_size[0] = local_work_size[1] = local_work_size[2] = 8;
//      }
//
//      if (texture_size.height==1) {
//        local_work_size[1] = 1;
//      }

      if (texture_size.width < local_work_size[0]) local_work_size[0] = texture_size.width;
      if (texture_size.height < local_work_size[1]) local_work_size[1] = texture_size.height;
      if (texture_size.depth < local_work_size[2]) local_work_size[2] = texture_size.depth;

      size_t global_work_size[3] = {
              ((texture_size.width + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0],
              ((texture_size.height + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1],
              ((texture_size.depth + local_work_size[2] - 1) / local_work_size[2]) * local_work_size[2]
      };

#ifdef PRINT_DEBUG
      std::cout << "Function "<<kernel_name_
                << " blocks: "
                << local_work_size[0] << "x" << local_work_size[1] << "x" << local_work_size[2]
                << " grid: "
                << global_work_size[0] << "x" << global_work_size[1] << "x" << global_work_size[2]
                << std::endl;
#endif

      cl_int last_error = 0;
      cl_event    waiting_event = nullptr;

      if (command_->get_wait_completed())
        waiting_event = clCreateUserEvent(command_->get_context(), &last_error);

      last_error = clEnqueueNDRangeKernel(command_->get_command_queue(),
                                          kernel_, 3,
                                          nullptr,
                                          global_work_size,
                                          nullptr,
                                          //local_work_size,
                                          0,
                                          nullptr,
                                          &waiting_event);

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to enqueue kernel: " + kernel_name_ + " error code: " + std::to_string(last_error));
      }

      if (waiting_event && command_->get_wait_completed()) {
        last_error = clWaitForEvents(1, &waiting_event);

        clReleaseEvent(waiting_event);

        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to waiting execution of kernel: " + kernel_name_ + " error code: " + std::to_string(last_error));
        }
      }
    }

    Function::Function(
            dehancer::opencl::Command *command,
            const std::string& kernel_name,
            const std::string &library_path):
            command_(command),
            kernel_name_(kernel_name),
            library_path_(library_path),
            kernel_(nullptr),
            encoder_(nullptr),
            arg_list_({})
    {
      std::unique_lock<std::mutex> lock(Function::mutex_);

      if (kernel_map_.find(command_->get_command_queue()) != kernel_map_.end())
      {
        auto& km =  kernel_map_[command_->get_command_queue()];
        if (km.find(kernel_name_) != km.end()) {
          kernel_ = km[kernel_name_];
          encoder_ = std::make_shared<opencl::CommandEncoder>(kernel_,this);
          return;
        }
      }
      else {
        kernel_map_[command_->get_command_queue()] = {};
      }

      cl_program program_ = nullptr;

      auto p_path = library_path_.empty() ? dehancer::device::get_lib_path() : library_path;
      std::size_t p_path_hash = std::hash<std::string>{}(p_path);

      std::string library_source_;
      if (p_path.empty()) {
        p_path_hash = dehancer::device::get_lib_source(library_source_);
        if (library_source_.empty())
          throw std::runtime_error("Could not find embedded opencl source code for '" + kernel_name + "'");
      }

      if (program_map_.find(command_->get_command_queue()) != program_map_.end())
      {
        auto& pm =  program_map_[command_->get_command_queue()];
        if (pm.find(p_path_hash) != pm.end()) {
          program_ = pm[p_path_hash];
        }
      }
      else {
        program_map_[command_->get_command_queue()] = {};
      }

      cl_int last_error = 0;

      if (program_ == nullptr) {
        std::string source;
        const char *source_str;
        size_t source_size = source.size();

        if (library_source_.empty()) {
          source = clHelper::getEmbeddedProgram(p_path);
          source_str = source.c_str();
          source_size = source.size();
        }
        else {
          source_str = library_source_.c_str();
          source_size = library_source_.size();
        }

        program_ = clCreateProgramWithSource(command_->get_context(), 1, (const char **) &source_str,
                                             (const size_t *) &source_size, &last_error);

        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to create OpenCL program from exampleKernel.cl");
        }

        /* Build Kernel Program */
        auto device_id = command_->get_device_id();
        last_error = clBuildProgram(program_, 1, &device_id, "-cl-std=CL1.2 -cl-kernel-arg-info", nullptr, nullptr);

        if (last_error != CL_SUCCESS) {

          std::string log = "Unable to build OpenCL program from: " + kernel_name_;

          if (last_error < 0) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program_, command_->get_device_id(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            //build_log_.resize(log_size);
            log.resize(log_size);
            // Get the log
            clGetProgramBuildInfo(program_, command_->get_device_id(), CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                                  NULL);
          }

          throw std::runtime_error("Unable to build OpenCL program from: " + kernel_name_ + ": \n" + log);
        }

        program_map_[command_->get_command_queue()][p_path_hash] = program_ ;
      }

      kernel_ = clCreateKernel(program_, kernel_name_.c_str(), &last_error);

      kernel_map_[command_->get_command_queue()][kernel_name_]=kernel_;

      if (last_error != CL_SUCCESS) {
        throw std::runtime_error("Unable to create kernel for: " + kernel_name_);
      }

      encoder_ = std::make_shared<opencl::CommandEncoder>(kernel_, this);

    }

    Function::~Function() = default;

    const std::vector<dehancer::Function::ArgInfo>& Function::get_arg_info_list() const {
      if (arg_list_.empty()) {
        cl_int arg_num = 0;
        cl_int last_error = clGetKernelInfo (	kernel_,
                                                 CL_KERNEL_NUM_ARGS,
                                                 sizeof(cl_uint),
                                                 &arg_num,
                                                 nullptr
        );
        if (last_error != CL_SUCCESS) {
          throw std::runtime_error("Unable to get kernel info for: " + kernel_name_);
        }
        for (int i = 0; i < arg_num; ++i) {
          char name[256];
          char type_name[256];
          last_error = clGetKernelArgInfo(
                  kernel_,
                  i,
                  CL_KERNEL_ARG_NAME,
                  sizeof(name),
                  name,
                  nullptr);
          if (last_error != CL_SUCCESS) {
            throw std::runtime_error("Unable to get kernel arg info for: " + kernel_name_);
          }
          last_error = clGetKernelArgInfo(
                  kernel_,
                  i,
                  CL_KERNEL_ARG_TYPE_NAME,
                  sizeof(type_name),
                  type_name,
                  nullptr);
          if (last_error != CL_SUCCESS) {
            throw std::runtime_error("Unable to get kernel arg info for: " + kernel_name_);
          }
          dehancer::Function::ArgInfo info = {
                  .name = name,
                  .index = static_cast<uint>(i),
                  .type_name = type_name
          };

          arg_list_.push_back(info);
        }
      }
      return arg_list_;
    }

    const std::string &Function::get_name() const {
      return kernel_name_;
    }
}
