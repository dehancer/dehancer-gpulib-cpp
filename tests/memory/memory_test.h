//
// Created by denn on 02.01.2021.
//

#pragma once
#include <string>

#include "dehancer/gpu/Lib.h"
#include "tests/test_config.h"

#include <cuda.h>

auto memory_test =  [] (int dev_num,
                        const void* command_queue,
                        const std::string& platform) {

    std::cout << "Test memory object on platform: " << platform << std::endl;

    int N = 1024;
    size_t size = N * sizeof(float );

    auto* h_A = new float[N];
    auto* h_B = new float[N];

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
      h_A[i] = i;
      h_B[i] = i%2;
    }

    auto kernel = dehancer::Function(command_queue, "kernel_vec_add");

    auto A = dehancer::MemoryDesc({
                                          .length = size
                                  }).make(command_queue, h_A);

    auto B = dehancer::MemoryDesc({
                                          .length = size
                                  }).make(command_queue, h_B);

    auto C = dehancer::MemoryDesc({
                                          .length = size
                                  }).make(command_queue);

    auto D = dehancer::MemoryDesc({
                                          .type = dehancer::MemoryDesc::MemType::device
                                  }).make(command_queue, C->get_memory());

    kernel.execute([N, &A, &B, &C](dehancer::CommandEncoder& command_encoder){
        command_encoder.set(A,0);
        command_encoder.set(B,1);
        command_encoder.set(C,2);
        command_encoder.set(N, 3);
        return dehancer::CommandEncoder::Size{(size_t)N,1,1};
    });

    std::vector<float> result; result.resize(N);

    C->get_contents(result.data(), result.size()*sizeof(float ));

    std::cout << "summ: " << std::endl;
    for (auto v: result) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    auto kernel_dev = dehancer::Function(command_queue, "kernel_vec_dev");

    kernel_dev.execute([N, &D](dehancer::CommandEncoder& command_encoder){
        command_encoder.set(D,0);
        command_encoder.set(N, 1);
        return dehancer::CommandEncoder::Size{(size_t)N,1,1};
    });

    D->get_contents(result.data(), result.size()*sizeof(float ));

    std::cout << "dev: " << std::endl;
    for (auto v: result) {
      std::cout << v << " ";
    }
    std::cout << std::endl;


    delete[] h_A;
    delete[] h_B;

    return 0;
};