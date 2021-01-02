//
// Created by denn nevera on 30/11/2020.
//

#pragma once

#include <vector>
#include <set>
#include <cmath>
#include <limits>
#include <cfloat>

namespace dehancer::math {

    void make_gaussian_kernel(std::vector<float>& kernel, size_t size, float sigma);

    /**
     * Weights/Offsets
     */
    using KernelLine = std::pair<std::vector<float>, std::vector<float>>;

    /**
     * Make optimized gaussian kernel
     * @param kernel
     * @param sigma
     * @param accuracy
     * @param maxRadius
     */
    void make_gaussian_kernel(KernelLine &kernel,
                              float sigma,
                              float accuracy,
                              int maxRadius = 50);

    ///
    /// http://blog.ivank.net/fastest-gaussian-blur.html
    ///
    /***
     * Make optimized integral gaussian boxes
     * @param boxes
     * @param sigma
     * @param box_number
     */
    void make_gaussian_boxes(std::vector<float>& boxes, float sigma, size_t box_number);
}
