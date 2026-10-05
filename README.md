# dehancer-gpulib-cpp

## Build and install

Depends on OpenCV, and installed `dehancer_common_cpp`, `dehancer_xmp_cpp`,
`dehancer_maths_cpp`. OpenCL additionally requires `dehancer_opencl_helper`;
CUDA and Metal require their universes as well.

**Note: gpulib is the source of truth for the GPU selection. All CMake consumer
projects are expected to use the GPU specified by gpulib.**

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DDEHANCER_GPU_METAL=ON
cmake --build build --parallel $(nproc)
cmake --install build --parallel $(nproc)
```

Select the appropriate GPU backend: `DEHANCER_GPU_CUDA`,
`DEHANCER_GPU_OPENCL` or `DEHANCER_GPU_METAL`.

Make sure to set proper `CMAKE_PREFIX_PATH` and `CMAKE_INSTALL_PREFIX` to
discover dependencies and install.

`CMAKE_POSITION_INDEPENDENT_CODE` is set to `ON`.

The library is always built static.

## Windows build

We build in [Git Bash](https://gitforwindows.org) with `clang-cl`
and we add a magic string to CMake to select runtime.

Use Ninja as a make file generator.

You will need certain dependencies from vcpkg installed.

You might need to force OpenCV to build statically.

Set `PATH` to include `clang-cl.exe` from VS.

```sh
export PATH="$PATH:/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin"
```

Add something like this to cmake configuration:

```
-G Ninja \
-DCMAKE_C_COMPILER=clang-cl \
-DCMAKE_CXX_COMPILER=clang-cl \
-DCMAKE_MSVC_RUNTIME_LIBRARY='MultiThreaded$<$<CONFIG:Debug>:Debug>' \
-DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
-DVCPKG_TARGET_TRIPLET=x64-windows-static \
-DVCPKG_APPLOCAL_DEPS=OFF \
-DOpenCV_STATIC=ON
```

## Usage in CMake

```cmake
find_package(dehancer_gpulib CONFIG REQUIRED)
target_link_libraries(app PRIVATE dehancer_gpulib::dehancer_gpulib)
```

### Exported variables

`DEHANCER_BACKEND_GPU_NAME` identifies the selected backend in the package
config, and it's one of:

* `metal`
* `cuda`
* `opencl`

It's also `#define`d in `include/dehancer/gpulib_version.h` for usage in source
codes and set as a env variable in `share/dehancer_gpulib/gpulib.sh` for usage
in build scripts.

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
