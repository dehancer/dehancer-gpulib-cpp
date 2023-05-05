//
// Created by denn nevera on 30/11/2020.
//

#pragma once
#if WIN32
//#define _USE_MATH_DEFINES // for C++
#endif
#include <vector>
#include <set>
#include <cmath>
#include <limits>
#include <cfloat>

namespace dehancer::math {
    /***
     * https://en.wikipedia.org/wiki/Gaussian_blur
     * @param kernel
     * @param size
     * @param sigma
     */
    void make_gaussian_kernel(std::vector<float>& kernel, size_t size, float sigma);
    
    /**
     * http://www.johncostella.com/magic/
     * @param length
     * @param kernel
     */
    void magic_resampler(float length, std::vector<float>& kernel);
}
