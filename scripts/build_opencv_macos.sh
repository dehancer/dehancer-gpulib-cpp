#!/usr/bin/env bash

HOME_PWD="$(pwd)"

NCPUS=$(sysctl -n hw.ncpu)

CMAKE_INSTALL_PREFIX="/usr/local/universal"

if command -v brew > /dev/null ; then
    if brew ls libpng > /dev/null ; then
        echo -e '\033[1;31mWarning: \033[0mlibpng installed via HomeBrew'
        read -p "[Enter to continue] "
    fi
fi

while [ $# -gt 0 ]; do
    case "$1" in
        "--prefix")
        CMAKE_INSTALL_PREFIX="$2"
        echo "Installation path: ${CMAKE_INSTALL_PREFIX}"
        shift
        ;;
    esac
    shift
done


if [ ! -d "/tmp/opencv/opencv/" ]; then
  cd /tmp && mkdir -p "opencv" && cd opencv || exit 1
  git clone --depth 1 --single-branch --progress --verbose -b 4.x https://github.com/opencv/opencv.git
  cd opencv || exit 1
else
  cd /tmp/opencv/opencv || exit 1
fi

mkdir -p build-macos-universal && cd build-macos-universal || exit 1

cmake -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DBUILD_PNG=ON -DBUILD_OPENEXR=ON -DBUILD_TIFF=ON -DBUILD_WEBP=ON \
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX} \
    -DBUILD_SHARED_LIBS=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DWITH_EIGEN=OFF\
    -DWITH_JPEG=ON -DBUILD_JPEG=ON \
    -DVIDEOIO_ENABLE_PLUGINS=ON -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a"\
    -DOPENCV_DNN_OPENCL=OFF -DCAROTENE_NEON_ARCH=OFF -DBUILD_opencv_dnn=OFF  \
    -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF  ..

cmake --build . -j"${NCPUS}"  --config Release
sudo cmake --build . -j"${NCPUS}"  --target install

