__kernel void grid_kernel(int levels, __write_only image2d_t destination )
{

  int w = get_image_width (destination);
  int h = get_image_height (destination);

  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  float2 coords = (float2)((float)gid.x / (w - 1),
                           (float)gid.y / (h - 1));

  int num = levels*2;
  int index_x = int(coords.x*(num));
  int index_y = int(coords.y*(num));

  int index = clamp((index_y+index_x)%2,int(0),int(num));

  float ret = (float)(index);

  float4 color = {ret*coords.x,ret*coords.y,ret,1.0} ;//ao_bench(nsubsamples, x, y, w, h);

  write_imagef(destination, gid, color);

}

__kernel void convolve_line_kernel(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int kRadius,
        __global float* kern,
        __global float* kernSum
) {

  sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  int2 gid = (int2)(get_global_id(0),
                    get_global_id(1));

  int2 imageSize = (int2)(get_image_width(destination),
                          get_image_height(destination));

  if (gid.x >= imageSize.x || gid.y >= imageSize.y)
  {
    return;
  }

  //float kern0 = kern[0];

  // Normalize coordinates
  float2 coords = (float2)((float)gid.x / (imageSize.x - 1),
                           (float)gid.y / (imageSize.y - 1));


  float4 inColor = read_imagef(source, sampler, coords);

  //int firstPart = kRadius < length ? kRadius : length;

  float4 color = inColor;
  //float4 result = color*kern0;

  write_imagef(destination, gid, color);
}

__kernel void box_blur_swap_kernel (__global float* scl,
                                    __global float* tcl,
                                    int w,
                                    int h,
                                    int r) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    tcl[index] = scl[index];
  }
}

__kernel void box_blur_horizontal_kernel (__global float* scl,
                                          __global float* tcl,
                                          int w,
                                          int h,
                                          int r) {
  float iarr = 1.0 / (float)(r+r+1);
  int i = get_global_id(1);
  if (i>=h) return ;
  int ti = i*w, li = ti, ri = ti+r;
  float fv = scl[ti], lv = scl[ti+w-1], val = (float)(r+1.0)*fv;
  for(int j=0; j<r; j++) val += scl[ti+j];
//  for(int j=0  ; j<=r ; j++) { val += scl[ri++] - fv       ;   tcl[ti++] = round(val*iarr); }
//  for(int j=r+1; j<w-r; j++) { val += scl[ri++] - scl[li++];   tcl[ti++] = round(val*iarr); }
//  for(int j=w-r; j<w  ; j++) { val += lv        - scl[li++];   tcl[ti++] = round(val*iarr); }
  for(int j=0  ; j<=r ; j++) { val += scl[ri++] - fv       ;   tcl[ti++] = val*iarr; }
  for(int j=r+1; j<w-r; j++) { val += scl[ri++] - scl[li++];   tcl[ti++] = val*iarr; }
  for(int j=w-r; j<w  ; j++) { val += lv        - scl[li++];   tcl[ti++] = val*iarr; }
}

__kernel void box_blur_vertical_kernel (__global float* scl,
                                        __global float* tcl,
                                        int w,
                                        int h,
                                        int r) {
  float iarr = 1.0 / (float)(r+r+1);
  int i = get_global_id(0);
  if (i>=w) return ;
  //for(var i=0; i<w; i++) {
  int ti = i, li = ti, ri = ti+r*w;
  float fv = scl[ti], lv = scl[ti+w*(h-1)], val = (r+1)*fv;
  for(int j=0; j<r; j++) val += scl[ti+j*w];
//  for(int j=0  ; j<=r ; j++) { val += scl[ri] - fv     ;  tcl[ti] = round(val*iarr);  ri+=w; ti+=w; }
//  for(int j=r+1; j<h-r; j++) { val += scl[ri] - scl[li];  tcl[ti] = round(val*iarr);  li+=w; ri+=w; ti+=w; }
//  for(int j=h-r; j<h  ; j++) { val += lv      - scl[li];  tcl[ti] = round(val*iarr);  li+=w; ti+=w; }
  for(int j=0  ; j<=r ; j++) { val += scl[ri] - fv     ;  tcl[ti] = val*iarr;  ri+=w; ti+=w; }
  for(int j=r+1; j<h-r; j++) { val += scl[ri] - scl[li];  tcl[ti] = val*iarr;  li+=w; ri+=w; ti+=w; }
  for(int j=h-r; j<h  ; j++) { val += lv      - scl[li];  tcl[ti] = val*iarr;  li+=w; ti+=w; }
  //}
}

__kernel void box_blur_horizontal_image_kernel (
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int r)
{

  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  int w = get_image_width(source);
  int h = get_image_height(source);
  float i = (float)get_global_id(1);
  float x = (float)get_global_id(0);

  float2 dim = (float2)1.0/(float2)(1,1);
  float iarr = 1.0 / ((float)(r+r)+1.0);

  int ti = i*w, li = ti, ri = ti+r;

  float2 coords = (float2)(x, ti)*dim;
  float4 fv = read_imagef(source, sampler, coords);

  coords = (float2)(x, ti+w-1)*dim;
  float4 lv = read_imagef(source, sampler, coords);

  float4 val = (float4)((float)r+1.0)*(float4)fv;

  for(int j=0; j<r; j++) {
    coords = (float2)(x, ti + j)*dim;
    val += read_imagef(source, sampler, coords);
  }

  for(int j=0; j<=r ; j++)
  {
    coords = (float2)(x, ri++)*dim;
    val += read_imagef(source, sampler, coords) - fv;
    int2 coords2 = (int2)(x, ti);
    write_imagef(destination, coords2, val*iarr);
  }


  for(int j=r+1.0; j<w-r; j++)
  {
    float2 coords1 = (float2)(x, ri++)*dim;
    float2 coords2 = (float2)(x, li++)*dim;
    val += read_imagef(source, sampler, coords1) - read_imagef(source, sampler, coords2);
    int2 coords3 = (int2)(x, ti++);
    write_imagef(destination, coords3, val*iarr);
  }

  for(int j=w-r; j<w ; j++)
  {
    float2 coords1 = (float2)(x,li++)*dim;
    val += lv - read_imagef(source, sampler, coords1);
    int2 coords2 = (int2)(x, ti++);
    write_imagef(destination, coords2, val*iarr);
  }
}


__kernel void box_blur_horizontal_image_transpose_kernel (
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int r)
{

  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  int w = get_image_width(destination)+r;
  int h = get_image_height(destination)+r;
  float i = (float)get_global_id(1);
  float x = (float)get_global_id(0);

  float2 dim = (float2)(1,1)/(float2)(1,1);
  float iarr = 1.0 / ((float)(r+r)+1.0);

  int ti = i*w, li = ti, ri = ti+r;

  float2 coords = (float2)(x, ti)*dim;
  float4 fv = read_imagef(source, sampler, coords);

  coords = (float2)(x, ti+w-1)*dim;
  float4 lv = read_imagef(source, sampler, coords);

  float4 val = (float4)((float)r+1.0)*(float4)fv;

  for(int j=0; j<r; j++) {
    coords = (float2)(x, ti + j)*dim;
    val += read_imagef(source, sampler, coords);
  }

  for(int j=0; j<=r ; j++)
  {
    coords = (float2)(x, ri++)*dim;
    val += read_imagef(source, sampler, coords) - fv;

    int2 coords2 = (int2)(ti,x);
    write_imagef(destination, coords2, val*iarr);
  }


  for(int j=r+1.0; j<w-r; j++)
  {
    float2 coords1 = (float2)(x, ri++)*dim;
    float2 coords2 = (float2)(x, li++)*dim;
    val += read_imagef(source, sampler, coords1) - read_imagef(source, sampler, coords2);

    int2 coords3 = (int2)(ti++,x);
    write_imagef(destination, coords3, val*iarr);
  }

  for(int j=w-r; j<w ; j++)
  {
    float2 coords1 = (float2)(x,li++)*dim;
    val += lv - read_imagef(source, sampler, coords1);

    int2 coords2 = (int2)(ti++,x);
    write_imagef(destination, coords2, val*iarr);
  }
}


__kernel void box_blur_vertical_image_kernel__ (
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int r)
{

  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  int w = get_image_width(source);
  int h = get_image_height(source);
  float i = (float)get_global_id(0);
  float y = (float)get_global_id(1);

  float2 dim = (float2)1.0/(float2)(1,1);
  float iarr = 1.0 / ((float)(r+r)+1.0);

  int ti = i*h, li = ti, ri = ti+r;

  float2 coords = (float2)(ti,y)*dim;
  float4 fv = read_imagef(source, sampler, coords);

  coords = (float2)(ti+h-1,y)*dim;
  float4 lv = read_imagef(source, sampler, coords);

  float4 val = (float4)((float)r+1.0)*(float4)fv;

  for(int j=0; j<r; j++) {
    coords = (float2)(ti + j,y)*dim;
    val += read_imagef(source, sampler, coords);
  }

  for(int j=0; j<=r ; j++)
  {
    coords = (float2)(ri++,y)*dim;
    val += read_imagef(source, sampler, coords) - fv;
    int2 coords2 = (int2)(ti,y);
    write_imagef(destination, coords2, val*iarr);
  }


  for(int j=r+1.0; j<h-r; j++)
  {
    float2 coords1 = (float2)(ri++,y)*dim;
    float2 coords2 = (float2)(li++,y)*dim;
    val += read_imagef(source, sampler, coords1) - read_imagef(source, sampler, coords2);
    int2 coords3 = (int2)(ti++,y);
    write_imagef(destination, coords3, val*iarr);
  }

  for(int j=h-r; j<h ; j++)
  {
    float2 coords1 = (float2)(li++,y)*dim;
    val += lv - read_imagef(source, sampler, coords1);
    int2 coords2 = (int2)(ti++,y);
    write_imagef(destination, coords2, val*iarr);
  }
}

//function boxBlurT_4 (scl, tcl, w, h, r) {
//  var iarr = 1 / (r+r+1);
//  for(var i=0; i<w; i++) {
//    var ti = i, li = ti, ri = ti+r*w;
//    var fv = scl[ti], lv = scl[ti+w*(h-1)], val = (r+1)*fv;
//    for(var j=0; j<r; j++) val += scl[ti+j*w];
//    for(var j=0  ; j<=r ; j++) { val += scl[ri] - fv     ;  tcl[ti] = Math.round(val*iarr);  ri+=w; ti+=w; }
//    for(var j=r+1; j<h-r; j++) { val += scl[ri] - scl[li];  tcl[ti] = Math.round(val*iarr);  li+=w; ri+=w; ti+=w; }
//    for(var j=h-r; j<h  ; j++) { val += lv      - scl[li];  tcl[ti] = Math.round(val*iarr);  li+=w; ti+=w; }
//  }
//}

__kernel void box_blur_vertical_image_kernel(
        __read_only image2d_t source,
        __write_only image2d_t destination,
        int r)
{

  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  int w = get_image_width(destination);
  int h = get_image_height(destination);
  int i = get_global_id(0);
  int y = get_global_id(1);

  float2 dim = (float2)(1.0,1.0);///(float2)(1,1);

  float iarr = 1.0 / (float )(r+r+1);

  //for(var i=0; i<w; i++) {
  int ti = i, li = ti, ri = ti+r;

  float2 coords = (float2)(ti,y)*dim;
  float4 fv = read_imagef(source, sampler, coords);

  coords = (float2)(ti+(h-1), y)*dim;
  float4 lv = read_imagef(source, sampler, coords);

  float4 val = (float4)(r+1)*(float4)fv;

  for(int j=0; j<r; j++) {
    coords = (float2)(ti+j,y)*dim;
    val += read_imagef(source, sampler, coords);
  }

  for(int j=0  ; j<=r ; j++) {
    coords = (float2)(ri++,y)*dim;

    val += read_imagef(source, sampler, coords) - fv; //scl[ri] - fv;

    int2 coords2 = (int2)(ti++,y);
    write_imagef(destination, coords2, val*iarr);

    //tcl[ti] = Math.round(val*iarr);

    //ri+=1;
    //ti+=1;
  }

  for(int j=r+1; j<h-r; j++) {
    float2 coords1 = (float2)(ri++,y)*dim;
    float2 coords2 = (float2)(li++,y)*dim;
    val += read_imagef(source, sampler, coords1) - read_imagef(source, sampler, coords2);//scl[ri] - scl[li];

    int2 coords3 = (int2)(ti++,y);
    write_imagef(destination, coords3, val*iarr);

    //tcl[ti] = Math.round(val*iarr);
    //li+=1; ri+=1; ti+=1;
  }

  for(int j=h-r; j<h  ; j++) {
    coords = (float2)(li++,y)*dim;
    val += lv - read_imagef(source, sampler, coords);//scl[li];
    int2 coords2 = (int2)(ti++,y);
    write_imagef(destination, coords2, val*iarr);
    //tcl[ti] = Math.round(val*iarr);
    //li+=1; ti+=1;
  }
  //}
}

__kernel void image_to_channels (
        __read_only image2d_t source,
        __global float* reds,
        __global float* greens,
        __global float* blues,
        __global float* alphas)
{
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(source);
  int h = get_image_height(source);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);

    float4 color     = read_imagef(source, sampler, gid);

    reds[index] = color.r;
    greens[index] = color.g;
    blues[index] = color.b;
    alphas[index] = color.rgba.a;
  }

}

__kernel void channels_to_image (
        __write_only image2d_t destination,
        __global float* reds,
        __global float* greens,
        __global float* blues,
        __global float* alphas)
{
  sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
  int x = get_global_id(0);
  int y = get_global_id(1);
  int w = get_image_width(destination);
  int h = get_image_height(destination);

  int2 gid = (int2)(x, y);

  if ((gid.x < w) && (gid.y < h)) {
    const int index = ((gid.y * w) + gid.x);
    float4 inColor = {reds[index], greens[index], blues[index], alphas[index]};
    write_imagef(destination, gid, inColor);
  }
}