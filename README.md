# dehancer-gpulib-cpp

## Build

Requires CMake 4.3+.

Depends on OpenCV, and installed `dehancer_common_cpp`, `dehancer_xmp_cpp`,
`dehancer_maths_cpp`. OpenCL additionally requires `dehancer_opencl_helper`;
CUDA and Metal require their universes as well.

gpulib is the source of truth regarding the GPU used. All consumers are expected
to follow the GPU as defined by gpulib.

```sh
# export PKG_CONFIG_PATH=/opt/dehancer-dependencies/lib/pkgconfig:/opt/dehancer-dependencies/lib64/pkgconfig
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$HOME/local-dehancer;/opt/dehancer-dependencies" \
  -DCMAKE_INSTALL_PREFIX="$HOME/local-dehancer" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DCREATE_PKG_CONFIG=OFF \
  -DDEHANCER_GPU_METAL=ON \
  -DDEHANCER_GPU_OPENCL=OFF \
  -DDEHANCER_GPU_CUDA=OFF
cmake --build build --config Release --parallel $(nproc)
cmake --install build --config Release
```

Select the appropriate `DEHANCER_GPU` backend on other platforms.

Adjacent dependencies reuse parent-provided targets first, then use
`find_package(... CONFIG REQUIRED)` through `CMAKE_PREFIX_PATH`. Missing packages
fail configuration; this project never fetches its dependencies. FetchContent
can fetch this library itself, but does not fetch its dependencies automatically.
Dependency fetching is centralized in `dehancer-video-universal`, enabled explicitly
with `DEHANCER_FETCH_DEPENDENCIES=ON`.

`CMAKE_INSTALL_LIBDIR` and `CMAKE_INSTALL_INCLUDEDIR` override standard installation
directories. Relative directories support relocation.

## Consume

```cmake
find_package(dehancer_gpulib CONFIG REQUIRED)
target_link_libraries(app PRIVATE dehancer_gpulib::dehancer_gpulib)
```

`DEHANCER_BACKEND_GPU_NAME` identifies the selected backend in the package config,
and it's one of:

* `metal`
* `cuda`
* `opencl`

It's also #defined in `include/dehancer/gpulib_version.h` and set as a env variable in
`share/dehancer_gpulib/gpulib.sh`.

Cmake files also export boolean definitions:

* `DEHANCER_GPU_CUDA`
* `DEHANCER_GPU_OPENCL`
* `DEHANCER_GPU_METAL`

Use `find_package(dehancer_gpulib CONFIG REQUIRED)` for every backend.
The same targets work with `add_subdirectory()` or:

```cmake
include(FetchContent)
FetchContent_Declare(dehancer_gpulib
    GIT_REPOSITORY https://github.com/dehancer/dehancer-gpulib-cpp.git
)
FetchContent_MakeAvailable(dehancer_gpulib)
target_link_libraries(app PRIVATE dehancer_gpulib::dehancer_gpulib)
```
