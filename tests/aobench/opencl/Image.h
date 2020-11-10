//
// Created by denn nevera on 06/11/2020.
//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

struct Image {

    Image(size_t _width, size_t _height)
            : width(_width),
              height(_height),
              length(4*width*height),
              pix(new float[length])
    {
    };

    ~Image() { delete[] pix; }

    size_t width, height, length;
    float *pix;

    void savePPM(const char *fName) const;
};
