#!/usr/bin/env bash

HOME_PWD="$(pwd)"

NCPUS=$(sysctl -n hw.ncpu)

CMAKE_INSTALL_PREFIX="/usr/local"

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

if [ ! -d "/tmp/jpeg-turbo/jpeg-turbo/" ]; then

  cd /tmp && mkdir -p "jpeg-turbo" && cd "jpeg-turbo" || exit 1
  git clone --depth 1 --single-branch --progress --verbose https://github.com/libjpeg-turbo/libjpeg-turbo.git --branch '3.0.x'
  cd "libjpeg-turbo" || exit 1

else
  cd "/tmp/jpeg-turbo/jpeg-turbo" || exit 1
fi


mkdir -p build-arm64 && cd build-arm64 && \
    cmake \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DWITH_JPEG8=1 \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"  \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC ..  \
    && make -j$(nproc) && sudo make install \
    && sudo rm -rf "${CMAKE_INSTALL_PREFIX}"/lib/libturbojpeg*.dylib \
    && sudo rm -rf "${CMAKE_INSTALL_PREFIX}"/lib/libjpeg*.dylib

cd ../

mkdir -p build-x86_64 && cd build-x86_64 && \
    cmake \
    -DCMAKE_OSX_ARCHITECTURES="x86_64" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DWITH_JPEG8=1 \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}/x86_64"  \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC ..  \
    && make -j$(nproc) && sudo make install \
    && sudo rm -rf "${CMAKE_INSTALL_PREFIX}"/lib/libturbojpeg*.dylib \
    && sudo rm -rf "${CMAKE_INSTALL_PREFIX}"/lib/libjpeg*.dylib

for entry in "${CMAKE_INSTALL_PREFIX}"/x86_64/lib/*.a
do
  name=$(basename -- "$entry")
  base="${CMAKE_INSTALL_PREFIX}"/lib/$name
  copy_name="$base".arm64
  sudo cp -nfi "$base" "$copy_name"
  echo "lipo $base && $entry"
  sudo lipo -create -output "$base" "$copy_name" "$entry"
  sudo rm -r "$copy_name" "$entry"
done

sudo rm -rf "${CMAKE_INSTALL_PREFIX}"/x86_64
