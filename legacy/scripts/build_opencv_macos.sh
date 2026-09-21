#!/usr/bin/env bash

HOME_PWD="$(pwd)"

NCPUS=$(sysctl -n hw.ncpu)

OPENCV_VERSION="4.10.0"
CMAKE_INSTALL_PREFIX="/usr/local/universal-${OPENCV_VERSION}"

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
  git clone --depth 1 --single-branch --progress --verbose -b "${OPENCV_VERSION}" https://github.com/opencv/opencv.git

  if [ ! -d "/tmp/opencv/opencv_contrib/" ]; then
    cd /tmp && mkdir -p "opencv" && cd opencv || exit 1
    git clone --depth 1 --single-branch --progress --verbose -b "${OPENCV_VERSION}" https://github.com/opencv/opencv_contrib.git
  fi

  cd opencv || exit 1

else
  cd /tmp/opencv/opencv || exit 1
fi


mkdir -p build-macos-arm64 && cd build-macos-arm64 || exit 1

cmake \
    -DCMAKE_OSX_ARCHITECTURES="arm64" \
    -DOPENCV_EXTRA_MODULES_PATH="/tmp/opencv/opencv_contrib/modules" \
    -DBUILD_opencv_legacy=OFF \
    -DBUILD_opencv_mcc=ON \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DWITH_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_JPEG=OFF \
    -DBUILD_PNG=OFF \
    -DBUILD_OPENEXR=ON \
    -DBUILD_TIFF=ON \
    -DBUILD_WEBP=ON \
    -DBUILD_OpenCV_HAL=OFF \
    -DBUILD_OPENVX=OFF \
    -DOBSENSOR_USE_ORBBEC_SDK=OFF \
    -DWITH_OBSENSOR=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH="${CMAKE_INSTALL_PREFIX}" \
    -DBUILD_SHARED_LIBS=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DWITH_EIGEN=OFF\
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
    -DVIDEOIO_ENABLE_PLUGINS=ON -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF ..

cmake --build . -j"${NCPUS}"  --config Release
sudo cmake --build . -j"${NCPUS}"  --target install

cd ../

mkdir -p build-macos-x86_64 && cd build-macos-x86_64 || exit 1

cmake \
    -DCMAKE_OSX_ARCHITECTURES="x86_64" \
    -DOPENCV_EXTRA_MODULES_PATH="/tmp/opencv/opencv_contrib/modules" \
    -DBUILD_opencv_legacy=OFF \
    -DBUILD_opencv_mcc=ON \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
    -DWITH_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_JPEG=OFF \
    -DBUILD_PNG=OFF \
    -DBUILD_OPENEXR=ON \
    -DBUILD_TIFF=ON \
    -DBUILD_WEBP=ON \
    -DBUILD_OpenCV_HAL=OFF \
    -DBUILD_OPENVX=OFF \
    -DOBSENSOR_USE_ORBBEC_SDK=OFF \
    -DWITH_OBSENSOR=OFF \
    -DCMAKE_INSTALL_PREFIX:PATH="${CMAKE_INSTALL_PREFIX}/x86_64" \
    -DBUILD_SHARED_LIBS=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DWITH_EIGEN=OFF\
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
    -DVIDEOIO_ENABLE_PLUGINS=ON -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF \
    -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF ..

cmake --build . -j"${NCPUS}"  --config Release
sudo cmake --build . -j"${NCPUS}"  --target install

for entry in "${CMAKE_INSTALL_PREFIX}"/x86_64/lib/opencv4/3rdparty/*.a
do
  name=$(basename -- "$entry")
  base="${CMAKE_INSTALL_PREFIX}"/lib/opencv4/3rdparty/$name
  copy_name="$base".arm64
  sudo cp -nfi "$base" "$copy_name"
  echo "lipo $base && $entry"
  sudo lipo -create -output "$base" "$copy_name" "$entry"
  sudo rm -r "$copy_name" "$entry"
done

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
