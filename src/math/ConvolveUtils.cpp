//
// Created by denn nevera on 03/12/2020.
//

#include "dehancer/gpu/math/ConvolveUtils.h"

#define DHCR_MAGIC_RESAMPLER_SHARPY 1

namespace dehancer::math {
    
    void make_gaussian_kernel(std::vector<float>& kernel, size_t size, float sigma){
      
      kernel.resize(size);
      
      int mean = floor((float )size / 2);
      float sum = 0; // For accumulating the kernel values
      for (int x = 0; x < size; x++)  {
        kernel[x] =  expf(-0.5f * powf((float )(x - mean) / sigma, 2.0));
        // Accumulate the kernel values
        sum += kernel[x];
      }
      
      for (int x = 0; x < size; x++)
        kernel[x] /= sum;
    }
    
    
    /**
    * Weights/Offsets
    */
    using KernelLine = std::pair<std::vector<float>, std::vector<float>>;
    
    void make_gaussian_kernel(KernelLine &kernel,
                              float sigma,
                              float accuracy,
                              int maxRadius) {
      int kRadius = static_cast<int>(std::ceil(sigma * std::sqrt(-2.0f * std::log(accuracy))) + 1.0f);
      if (maxRadius < 16) maxRadius = 16;         // too small maxRadius would result in inaccurate sum.
      if (kRadius > maxRadius) kRadius = maxRadius;
      
      kernel.first.resize(kRadius);
      kernel.second.resize(kRadius);
      
      for (int i = 0; i < kRadius; i++)   // Gaussian function
        kernel.first[i] = (float) (std::exp(-0.5 * i * i / sigma / sigma));
      
      if (kRadius < maxRadius && kRadius > 3) {   // edge correction
        float sqrtSlope = FLT_MAX;
        int r = kRadius;
        while (r > kRadius / 2) {
          r--;
          float a = std::sqrt(kernel.first[r]) / static_cast<float>((kRadius - r));
          if (a < sqrtSlope)
            sqrtSlope = a;
          else
            break;
        }
        for (int r1 = r + 2; r1 < kRadius; r1++)
          kernel.first[r1] = (float) ((kRadius - r1) * (kRadius - r1) * sqrtSlope * sqrtSlope);
      }
      
      float sum = 0; // sum over all kernel elements for normalization
      if (kRadius < maxRadius) {
        sum = kernel.first[0];
        for (int i = 1; i < kRadius; i++)
          sum += 2 * kernel.first[i];
      } else
        sum = sigma * sqrtf(2.0f * M_PI);
      
      float rsum = 0.5f + 0.5f * kernel.first[0] / sum;
      for (int i = 0; i < kRadius; i++) {
        float v = (kernel.first[i] / sum);
        kernel.first[i] = (float) v;
        rsum -= v;
        kernel.second[i] = (float) rsum;
      }
    }
    
    
    ///
    /// http://blog.ivank.net/fastest-gaussian-blur.html
    ///
    /***
     * Make optimized integral gaussian boxes
     * @param boxes
     * @param sigma
     * @param box_number
     */
    void make_gaussian_boxes(std::vector<float> &boxes, float sigma, size_t box_number) {
      auto n = static_cast<float>(box_number);
      float coeff = 12.0f;
      float wIdeal = sqrtf((coeff * sigma * sigma / n) + 1.0f);  // Ideal averaging filter width
      int wl = std::floor(wIdeal);
      if (wl % 2 == 0) wl--;
      int wu = wl + 2;
      
      float mIdeal = (coeff * sigma * sigma
                      - n * static_cast<float>(wl * wl)
                      - 4.0f * n * static_cast<float>(wl)
                      - 3.0f * n) / (-4.0f * static_cast<float>(wl) - 4.0f);
      
      int m = static_cast<int>(std::round(mIdeal));
      
      // var sigmaActual = Math.sqrt( (m*wl*wl + (n-m)*wu*wu - n)/12 );
      for (int i = 0; i < box_number; i++)
        boxes.push_back(static_cast<float>(i < m ? wl : wu));
    }
    
    void magic_resampler(float length, std::vector<float>& kernel) {
      int size = std::ceil(3.0f/2.0f*length);
      int half_size = (int)std::ceil((float)size/2.0f);
      
      if(half_size%2==0) half_size+=1;
      float sum = 0;
      for (int i = -half_size; i <= half_size; ++i) {
        
        auto x = (float )i;
        #ifdef DHCR_MAGIC_RESAMPLER_SHARPY
        if      ( x <  -3.0f/2.0f*length ) x = 0;
        else if ( x >= -3.0f/2.0f*length && x <  -1.0f/2.0f*length ) x = 1.0f/2.0f*pow(x+3.0/2.0*length,2.0f);
        else if ( x >= -1.0/2.0*length   && x <=  1.0f/2.0f*length ) x = 1.0f/2.0f*pow(x-4.0/3.0*length,2.0f);
        else if ( x >   1.0/2.0*length   && x <=  3.0f/2.0f*length ) x = 1.0f/2.0f*pow(x-3.0/2.0*length,2.0f);
        else if ( x >  -3.0f/2.0f*length ) x = 0;
        #else
        if      ( x <  -3.0f/2.0f*length ) x = 0;
        else if ( x >= -3.0f/2.0f*length && x <=0 ) x = 1.0f/2.0f*pow(x+3.0/2.0*length,2.0f);
        else if ( x >  0   && x <=  3.0f/2.0f*length ) x = 1.0f/2.0f*pow(x-3.0/2.0*length,2.0f);
        else if ( x >  -3.0f/2.0f*length ) x = 0;
        #endif
        kernel.push_back(x);
        sum += x;
      }
      
      for (auto& v: kernel) {
        v/=sum;
      }
    }
}