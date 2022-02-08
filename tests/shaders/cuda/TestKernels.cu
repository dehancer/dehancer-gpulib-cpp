//
// Created by denn nevera on 16/11/2020.
//

#include <cuda.h>
#include <cuda_fp16.h>

#include "TestKernels.h"

DHCR_KERNEL void kernel_half_test (
        texture2d_read_t tex       DHCR_BIND_TEXTURE(0),
        DHCR_CONST_ARG int_ref_t n DHCR_BIND_BUFFER(1),
        DHCR_CONST_ARG int_ref_t m DHCR_BIND_BUFFER(2)
        DHCR_KERNEL_GID_2D
        )
{
  float4 val = make_float4(0.0f);
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      val = tex.read(make_float2((float)col + 0.5f, (float)row + 0.5f));//tex2D(tex, col + 0.5f, row + 0.5f);
      printf ("% 15.8e, % 15.8e, % 15.8e | ", val.x, val.y, val.z);
    }
    printf ("\n");
  }
}