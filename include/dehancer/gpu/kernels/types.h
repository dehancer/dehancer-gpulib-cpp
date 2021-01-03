//
// Created by denn on 03.01.2021.
//

#ifndef DEHANCER_GPULIB_TYPES_H
#define DEHANCER_GPULIB_TYPES_H

typedef struct  {
    int gid;
    int size;
} Texel1d;

typedef struct  {
    int2 gid;
    int2 size;
} Texel2d;

typedef struct  {
    int3 gid;
    int3 size;
} Texel3d;

#endif //DEHANCER_GPULIB_TYPES_H
