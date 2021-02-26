//
// Created by denn nevera on 06/11/2020.
//


// -*- mode: c++ -*-
// ======================================================================== //
// Copyright 2017 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/* originally imported from https://github.com/ispc/ispc, under the
   following license */
/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  * Neither the name of Intel Corporation nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.


  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*
  Based on Syoyo Fujita's aobench: http://code.google.com/p/aobench
*/

#ifndef DEHANCER_OPENCL_HELPER_AOBENCHKERNEL_H
#define DEHANCER_OPENCL_HELPER_AOBENCHKERNEL_H

/*! special thanks to
  - Syoyo Fujita, who apparently created the first aobench that was the root of all this!
  - Matt Pharr, who's aoBench variant this is based on
  - the OSPRay Project, and in particular Johannes Guenther, for the random number generator
 */


#include "aoBench.h"
#include "dehancer/gpu/kernels/types.h"
#include "dehancer/gpu/kernels/constants.h"

inline DHCR_DEVICE_FUNC void rng_seed(DHCR_THREAD_ARG struct RNGState *rng, int s)
{
  const int a = 16807;
  const int q = 127773;
  const int r = 2836;
  
  if (s == 0) rng->seed = 1;
  else rng->seed = s & 0x7FFFFFFF;
  
  for (int j = TABLE_SIZE+WARMUP_ITERATIONS; j >= 0; j--) {
    int k = rng->seed / q;
    rng->seed = a*(rng->seed - k*q) - r*k;
    rng->seed = rng->seed & 0x7FFFFFFF;
    if (j < TABLE_SIZE) rng->table[j] = rng->seed;
  }
  rng->state = rng->table[0];
}

inline DHCR_DEVICE_FUNC float rng_getInt(DHCR_THREAD_ARG struct RNGState *rng)
{
  const int a = 16807;
  const int q = 127773;
  const int r = 2836;
  const int f = 1 + (2147483647 / TABLE_SIZE);
  
  int k = rng->seed / q;
  rng->seed = a*(rng->seed - k*q) - r*k;
  rng->seed = rng->seed & 0x7FFFFFFF;
  int j = min(rng->state / f, TABLE_SIZE-1);
  rng->state = rng->table[j];
  rng->table[j] = rng->seed;
  return rng->state;
}

inline DHCR_DEVICE_FUNC float rng_getFloat(DHCR_THREAD_ARG struct RNGState *rng)
{
  return rng_getInt(rng) / 2147483647.0f;
}

inline DHCR_DEVICE_FUNC float dot3f(struct vec3f a, struct vec3f b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline DHCR_DEVICE_FUNC struct vec3f cross3f(struct vec3f v0, struct vec3f v1) {
  struct vec3f ret;
  ret.x = v0.y * v1.z - v0.z * v1.y;
  ret.y = v0.z * v1.x - v0.x * v1.z;
  ret.z = v0.x * v1.y - v0.y * v1.x;
  return ret;
}

inline DHCR_DEVICE_FUNC struct vec3f mul3ff(struct vec3f v, float f)
{
  struct vec3f ret;
  ret.x = v.x * f;
  ret.y = v.y * f;
  ret.z = v.z * f;
  return ret;
}

inline DHCR_DEVICE_FUNC struct vec3f add3f (struct vec3f a, struct vec3f b)
{
  struct vec3f ret;
  ret.x = a.x+b.x;
  ret.y = a.y+b.y;
  ret.z = a.z+b.z;
  return ret;
}

inline DHCR_DEVICE_FUNC struct vec3f sub3f (struct vec3f a, struct vec3f b)
{
  struct vec3f ret;
  ret.x = a.x-b.x;
  ret.y = a.y-b.y;
  ret.z = a.z-b.z;
  return ret;
}

inline DHCR_DEVICE_FUNC struct vec3f madd3ff(struct vec3f a, float f, struct vec3f b)
{
  struct vec3f ret;
  ret.x = a.x + f * b.x;
  ret.y = a.y + f * b.y;
  ret.z = a.z + f * b.z;
  return ret;
}

inline DHCR_DEVICE_FUNC struct vec3f normalize3f(struct vec3f v)
{
  float len2 = dot3f(v, v);
  float invLen = rsqrt(len2);
  return mul3ff(v,invLen);
}

inline DHCR_DEVICE_FUNC void ray_plane_intersect(DHCR_THREAD_ARG struct Isect *isect, struct Ray ray, struct Plane plane)
{
  float d = -dot3f(plane.p, plane.n);
  float v =  dot3f(ray.dir, plane.n);
  
  if (fabs(v) < 1.0e-17f)
    return;
  else {
    float t = -(dot3f(ray.org, plane.n) + d) / v;
    
    if ((t > 0.0f) && (t < isect->t)) {
      isect->t = t;
      isect->hit = 1;
      isect->p = madd3ff(ray.org,t,ray.dir);
      isect->n = plane.n;
    }
  }
}


inline DHCR_DEVICE_FUNC void ray_sphere_intersect(DHCR_THREAD_ARG struct Isect *isect, struct Ray ray, struct Sphere sphere)
{
  struct vec3f rs = sub3f(ray.org,sphere.center);
  
  float B = dot3f(rs, ray.dir);
  float C = dot3f(rs, rs) - sphere.radius * sphere.radius;
  float D = B * B - C;
  
  if (D > 0.f) {
    float t = -B - sqrt(D);
    
    if ((t > 0.0f) && (t < isect->t)) {
      isect->t = t;
      isect->hit = 1;
      isect->p = madd3ff(ray.org,t,ray.dir);
      isect->n = normalize3f(sub3f(isect->p, sphere.center));
    }
  }
}


inline DHCR_DEVICE_FUNC void orthoBasis(struct vec3f basis[3], struct vec3f n)
{
  basis[2] = n;
  basis[1].x = 0.0f;
  basis[1].y = 0.0f;
  basis[1].z = 0.0f;
  
  if ((n.x < 0.6f) && (n.x > -0.6f)) {
    basis[1].x = 1.0f;
  } else if ((n.y < 0.6f) && (n.y > -0.6f)) {
    basis[1].y = 1.0f;
  } else if ((n.z < 0.6f) && (n.z > -0.6f)) {
    basis[1].z = 1.0f;
  } else {
    basis[1].x = 1.0f;
  }
  
  basis[0] = normalize3f(cross3f(basis[1], basis[2]));
  basis[1] = normalize3f(cross3f(basis[2], basis[0]));
}


static DHCR_DEVICE_FUNC inline float ambient_occlusion(DHCR_THREAD_ARG struct Isect *isect,
                                                           struct Plane plane, struct Sphere spheres[3],
                                                           DHCR_THREAD_ARG struct RNGState *rngstate) {
  float eps = 0.0001f;
  struct vec3f p;
  struct vec3f basis[3];
  float occlusion = 0.0f;
  
  p = madd3ff(isect->p,eps,isect->n);
  
  orthoBasis(basis, isect->n);
  
  const int ntheta = NAO_SAMPLES;
  const int nphi   = NAO_SAMPLES;
  #pragma unroll
  for (int j = 0; j < ntheta; j++) {
    #pragma unroll
    for (int i = 0; i < nphi; i++) {
      struct Ray ray;
      struct Isect occIsect;
      
      float theta = sqrt(rng_getFloat(rngstate));
      float phi   = 2.0f * M_PI * rng_getFloat(rngstate);
      float x = cos(phi) * theta;
      float y = sin(phi) * theta;
      float z = sqrt(1.0f - theta * theta);
      
      // local . global
      float rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
      float ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
      float rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;
      
      ray.org = p;
      ray.dir.x = rx;
      ray.dir.y = ry;
      ray.dir.z = rz;
      
      occIsect.t   = 1.0e+17f;
      occIsect.hit = 0;
      #pragma unroll
      for (int snum = 0; snum < 3; ++snum)
        ray_sphere_intersect(&occIsect, ray, spheres[snum]);
      ray_plane_intersect (&occIsect, ray, plane);
      
      if (occIsect.hit) occlusion += 1.0f;
    }
  }
  
  occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);
  return occlusion;
}

inline DHCR_DEVICE_FUNC float4 ao_bench(int nsubsamples, int x, int y, int w, int h) {
  
  struct Plane plane = { { 0.0f, -0.5f, 0.0f }, { 0.f, 1.f, 0.f } };
  struct Sphere spheres[3] = {
          { { -2.0f, 0.0f, -3.5f }, 0.5f },
          { { -0.5f, 0.0f, -3.0f }, 0.5f },
          { { 1.0f, 0.0f, -2.2f }, 0.5f } };
  struct RNGState rngstate;
  
  float invSamples = 1.f / nsubsamples;
  
  int offset = 4 * (y * w + x);
  
  rng_seed(&rngstate,offset);
  
  float ret = 0.f;
  #pragma unroll
  for (int v=0;v<nsubsamples;v++)
          #pragma unroll
          for (int u=0;u<nsubsamples;u++) {
            float du = (float)u * invSamples, dv = (float)v * invSamples;
            
            // Figure out x,y pixel in NDC
            float px =  (x + du - (w / 2.0f)) / (w / 2.0f);
            float py = -(y + dv - (h / 2.0f)) / (h / 2.0f);
            struct Ray ray;
            struct Isect isect;
            
            ray.org.x = 0.f;
            ray.org.y = 0.f;
            ray.org.z = 0.f;
            
            // Poor man's perspective projection
            ray.dir.x = px;
            ray.dir.y = py;
            ray.dir.z = -1.0f;
            ray.dir = normalize3f(ray.dir);
            
            isect.t   = 1.0e+17f;
            isect.hit = 0;
  
            #pragma unroll
            for (int snum = 0; snum < 3; ++snum)
              ray_sphere_intersect(&isect, ray, spheres[snum]);
            ray_plane_intersect(&isect, ray, plane);
            
            if (isect.hit) {
              ret += ambient_occlusion(&isect, plane, spheres, &rngstate);
            }
          }
  
  ret *= (invSamples * invSamples);

#ifdef __METAL_VERSION__
  return  {ret,ret,ret,1};
#else
  return  (float4){ret,ret,ret,1};
#endif
}

#endif //DEHANCER_OPENCL_HELPER_AOBENCHKERNEL_H
