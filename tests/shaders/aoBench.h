//
// Created by denn nevera on 22/10/2020.
//

#ifndef CLHELPER_TOOLS_AOBENCH_H
#define CLHELPER_TOOLS_AOBENCH_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#else
#define thread
#endif


#define NAO_SAMPLES		8
#define M_PI 3.1415926535f

/* random number generator, taken from the ospray project, http://www.ospray.org

   Special thanks to Johannes Guenther who originally added this neat
   rng to ospray!
 */
#define TABLE_SIZE 32
#define WARMUP_ITERATIONS 7
struct RNGState {
    int seed;
    int state;
    int table[TABLE_SIZE];
};

struct vec3f {
    float x,y,z;
};

struct Isect {
    float      t;
    struct vec3f p;
    struct vec3f n;
    int        hit;
};

struct Sphere {
    struct vec3f center;
    float      radius;
};

struct Plane {
    struct vec3f    p;
    struct vec3f    n;
};

struct Ray {
    struct vec3f org;
    struct vec3f dir;
};

#endif //CLHELPER_TOOLS_AOBENCH_H
