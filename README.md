# dehancer-gpulib-cpp

## Build and install

Depends on OpenCV, and installed `dehancer_common_cpp`, `dehancer_xmp_cpp`,
`dehancer_maths_cpp`. OpenCL additionally requires `dehancer_opencl_helper`;
CUDA and Metal require their universes as well.

**Note: gpulib is the source of truth for the GPU selection. All CMake consumer
projects are expected to use the GPU specified by gpulib.**

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DDEHANCER_GPU_METAL=ON \
  -DDEHANCER_GPU_OPENCL=OFF \
  -DDEHANCER_GPU_CUDA=OFF
cmake --build build --parallel $(nproc)
cmake --install build --parallel $(nproc)
```

Select the appropriate `DEHANCER_GPU_*` backend on other platforms.

Make sure to set proper `CMAKE_PREFIX_PATH` and `CMAKE_INSTALL_PREFIX` to discover dependencies and install.

`CMAKE_POSITION_INDEPENDENT_CODE` is set to `ON`.

The library is always built static.

## Usage in CMake

```cmake
find_package(dehancer_gpulib CONFIG REQUIRED)
target_link_libraries(app PRIVATE dehancer_gpulib::dehancer_gpulib)
```

### Exported variables

`DEHANCER_BACKEND_GPU_NAME` identifies the selected backend in the package config,
and it's one of:

* `metal`
* `cuda`
* `opencl`

It's also `#define`d in `include/dehancer/gpulib_version.h` for usage in source codes
and set as a env variable in `share/dehancer_gpulib/gpulib.sh` for usage in build scripts.

CMake files also export boolean definitions:

* `DEHANCER_GPU_CUDA`
* `DEHANCER_GPU_OPENCL`
* `DEHANCER_GPU_METAL`

## Usage with pkg-config

Disabled by default. Configure with `-DCREATE_PKG_CONFIG=ON` to generate and
install `dehancer-gpulib-cpp.pc`.

```sh
export PKG_CONFIG_PATH="$HOME/local-dehancer/lib/pkgconfig"
pkg-config --cflags --libs dehancer-gpulib-cpp
```

## Testing

Install GoogleTest. Then:

```sh
cmake -B build -DBUILD_TESTING=ON
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure
```
