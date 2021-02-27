//
// Created by denn on 18.01.2021.
//

#ifndef DEHANCER_GPULIB_BLEND_H
#define DEHANCER_GPULIB_BLEND_H

#include "dehancer/gpu/kernels/type_cast.h"
#include "dehancer/gpu/kernels/constants.h"

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) min_component(float2 v) {
  return min(v.x, v.y);
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) min_component(float3 v) {
  return min(v.x, min(v.y, v.z));
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) min_component(float4 v) {
  float2 v2 = fminf(make_float2(v.x, v.y), make_float2(v.z, v.w));
  return min(v2.x, v2.y);
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) max_component(float2 v) {
  return max(v.x, v.y);
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) max_component(float3 v) {
  return max(v.x, max(v.y, v.z));
}

inline DHCR_DEVICE_FUNC float __attribute__((overloadable)) max_component(float4 v) {
  float2 v2 = fmaxf(make_float2(v.x, v.y), make_float2(v.z, v.w));
  return max(v2.x, v2.y);
}

inline DHCR_DEVICE_FUNC float3 clipcolor_wlum(float3 c, float wlum) {
  
  float l = wlum;
  float n = min_component(c);
  float x = max_component(c);
  
  if (n < 0.0f) {
    float v = 1.0f/(l - n);
    c.x = l + ((c.x - l) * l) * v;
    c.y = l + ((c.y - l) * l) * v;
    c.z = l + ((c.z - l) * l) * v;
  }
  if (x > 1.0f) {
    float v = 1.0f/(x - l);
    c.y = l + ((c.x - l) * (1.0f - l)) * v;
    c.y = l + ((c.y - l) * (1.0f - l)) * v;
    c.z = l + ((c.z - l) * (1.0f - l)) * v;
  }
  
  return c;
}

inline DHCR_DEVICE_FUNC float3 clipcolor(float3 c) {
  float l = lum(c);
  float n = min(min(c.x, c.y), c.z);
  float x = max(max(c.x, c.y), c.z);
  
  if (n < 0.0f) {
    c.x = l + ((c.x - l) * l) / (l - n);
    c.y = l + ((c.y - l) * l) / (l - n);
    c.z = l + ((c.z - l) * l) / (l - n);
  }
  if (x > 1.0f) {
    c.x = l + ((c.x - l) * (1.0 - l)) / (x - l);
    c.y = l + ((c.y - l) * (1.0 - l)) / (x - l);
    c.z = l + ((c.z - l) * (1.0 - l)) / (x - l);
  }
  
  return c;
}

inline DHCR_DEVICE_FUNC float3 setlum(float3 c, float l) {
  float d = l - lum(c);
  c = c + make_float3(d);
  return clipcolor(c);
}


inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) blend_normal(float4 base, float4 overlay){
  
  float4 c2 = base;
  float4 c1 = overlay;
  
  float4 outputColor;
  
  float a = c1.w + c2.w * (1.0f - c1.w);
  float alphaDivisor = a + step(a, 0.0f); // Protect against a divide-by-zero blacking out things in the output
  
  outputColor.x = (c1.x * c1.w + c2.x * c2.w * (1.0f - c1.w))/alphaDivisor;
  outputColor.y = (c1.y * c1.w + c2.y * c2.w * (1.0f - c1.w))/alphaDivisor;
  outputColor.z = (c1.z * c1.w + c2.z * c2.w * (1.0f - c1.w))/alphaDivisor;
  outputColor.w = a;
  
  return clamp(outputColor, make_float4(-0.0f), make_float4(+1.0f));
}

inline  DHCR_DEVICE_FUNC float4 blend_luminosity(float4 baseColor, float4 overlayColor)
{
  float3 base_rgb = make_float3(baseColor);
  float3 over_rgb = make_float3(overlayColor);
  return make_float4(
          base_rgb
          *
          make_float3(1.0f - overlayColor.w)
          +
          setlum(base_rgb,
                 lum(over_rgb)
          )
          * overlayColor.w, baseColor.w);
}

inline  DHCR_DEVICE_FUNC float4 blend_overlay(float4 base, float4 overlay)
{
  float ra;
  if (2.0f * base.x < base.w) {
    ra = 2.0f * overlay.x * base.x + overlay.x * (1.0f - base.w) + base.x * (1.0f - overlay.w);
  } else {
    ra = overlay.w * base.w - 2.0f * (base.w - base.x) * (overlay.w - overlay.x) + overlay.x * (1.0f - base.x) + base.x * (1.0f - overlay.w);
  }
  
  float ga;
  if (2.0f * base.y < base.w) {
    ga = 2.0f * overlay.y * base.y + overlay.y * (1.0f - base.w) + base.y * (1.0f - overlay.w);
  } else {
    ga = overlay.w * base.w - 2.0f * (base.w - base.y) * (overlay.w - overlay.y) + overlay.y * (1.0f - base.w) + base.y * (1.0f - overlay.w);
  }
  
  float ba;
  if (2.0f * base.z < base.w) {
    ba = 2.0f * overlay.z * base.z + overlay.z * (1.0f - base.w) + base.z * (1.0f - overlay.w);
  } else {
    ba = overlay.w * base.w - 2.0f * (base.w - base.z) * (overlay.w - overlay.z) + overlay.z * (1.0f - base.w) + base.z * (1.0f - overlay.w);
  }
  
  return make_float4(ra, ga, ba, 1.0f);
}

inline DHCR_DEVICE_FUNC float4 blend_color(float4 base, float4 overlay){
  float3 base_rgb = make_float3(base);
  float3 overlay_rgb = make_float3(overlay);
  return make_float4(base_rgb * make_float3(1.0f - overlay.w) + setlum(overlay_rgb, lum(base_rgb)) * overlay.w, base.w);
}

inline DHCR_DEVICE_FUNC float4 blend_add(float4 base, float4 overlay){
  float3 base_rgb = make_float3(base);
  float3 overlay_rgb = make_float3(overlay);
  return clamp(make_float4(mix(base_rgb,
                               clamp(base_rgb+overlay_rgb, 0.0f, 1.0f),
                               make_float3(overlay.w)),1.0f), make_float4(0.0f), make_float4(1.0f));
}

inline DHCR_DEVICE_FUNC float4 blend_subtract(float4 base, float4 overlay){
  float3 base_rgb = make_float3(base);
  float3 overlay_rgb = make_float3(overlay);
  return clamp(make_float4(mix(base_rgb,
                               clamp(base_rgb-overlay_rgb, 0.0f, 1.0f),
                               make_float3(overlay.w)),1.0f), make_float4(0.0f), make_float4(1.0f));
}

inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) blend(float4 base, float4 overlay, DHCR_BlendingMode mode, float4 opacity){
  
  float3 base_opacity = clamp(make_float3(opacity), 0.0f, 1.0f);
  
  float3 overlay_rgb  = make_float3(overlay);
  float4 result = make_float4(overlay_rgb, opacity.w);
  
  switch (mode) {
    case DHCR_Luminosity:
      result = blend_luminosity(base, result);
      break;
    
    case DHCR_Color:
      result = blend_color(base, result);
      break;
    
    case DHCR_Normal:
      result = blend_normal(base, result);
      break;
  
    case DHCR_Overlay:
      result = blend_overlay(base, result);
      break;
  
    case DHCR_Mix:
      result = mix(base, overlay, opacity.w);
      break;
      
    case DHCR_Min:
      result = mix(base, fminf(overlay,base), opacity.w);
      break;
      
    case DHCR_Max:
      result = mix(base, fmaxf(overlay,base), opacity.w);
      break;
      
    case DHCR_Add:
      result = blend_add(base, result);
      break;
  
    case DHCR_Subtract:
      result = blend_subtract(base, result);
      break;
  }
  
  result = mix(base, result, make_float4(base_opacity, 1.0f));
  
  return  result;
}

inline DHCR_DEVICE_FUNC float4 __attribute__((overloadable)) blend(float4 base, float4 overlay, DHCR_BlendingMode mode, float opacity) {
  return blend(base,overlay,mode,make_float4(opacity));
}

inline DHCR_DEVICE_FUNC float3 __attribute__((overloadable)) blend(float3 base, float3 overlay, DHCR_BlendingMode mode, float opacity){
  float4 base_ = make_float4(base,1.0f);
  float4 overlay_ = make_float4(overlay,1.0f);
  return make_float3(blend(base_, overlay_, mode, opacity));
}


#endif //DEHANCER_GPULIB_BLEND_H
