Build M1
==========
    mkdir build-arm64 && cd build-arm64
    cmake -DPRINT_DEBUG=ON -DBUILD_TESTING=ON -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DDEHANCER_TARGET_ARCH=arm64-apple-macos11 -DDEHANCER_GPU_OPENCL=ON \
    -DDEHANCER_GPU_METAL=OFF -DDEHANCER_GPU_CUDA=OFF ..

Build Intel
==========
    mkdir build-x86_64 && cd build-x86_64
    cmake -DPRINT_DEBUG=ON -DBUILD_TESTING=ON \
    -DCMAKE_OSX_ARCHITECTURES=x86_64 -DDEHANCER_TARGET_ARCH=x86_64-apple-macos10.14 \
    -DDEHANCER_GPU_OPENCL=OFF -DDEHANCER_GPU_METAL=ON -DDEHANCER_GPU_CUDA=OFF ..


Build Windows10 x64
===================
    # CUDA
    cmake -G
    "Ninja"
    -DCMAKE_VERBOSE_MAKEFILE=ON
    -DDEHANCER_GPU_CUDA=ON
    -DDEHANCER_GPU_OPENCL=OFF
    -DBUILD_TESTING=ON
    -DPRINT_DEBUG=ON
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
    -DVCPKG_TARGET_TRIPLET=x64-windows-static 
    ..

Requirements
===========
    Nasm:
    brew install nasm
    or
    yum install nasm

Requirements Windows
=======
    1. xxd from msys2
    
    2. cuda toolkit: 
    https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

LibJPEG Turbo
=============

    git clone https://github.com/libjpeg-turbo/libjpeg-turbo
    cd libjpeg-turbo && mkdir build && cd build
    cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
    make -j4 && make install 

Centos7 (based for DaVinci Resolve 16) 
============
    yum install llvm-toolset-7
    scl enable llvm-toolset-7 bash
    cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_TESTING=ON ..
    
Ubuntu (20.04)
=============

    # add deb http://apt.llvm.org/hirsute/ llvm-toolchain-hirsute-11 main
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421
    sudo apt update
    sudo apt upgrade
    sudo apt install clang-11 clang-tools-11 clang-11-doc libclang-common-11-dev libclang-11-dev libclang1-11 clang-format-11  clangd-11
    sudo apt-get install libllvm-11-ocaml-dev libllvm11 llvm-11 llvm-11-dev llvm-11-doc llvm-11-examples llvm-11-runtime
    sudo apt-get install lldb-11
    sudo apt-get install lld-11
    sudo apt-get install libc++-11-dev libc++abi-11-dev
    sudo apt-get install libomp-11-dev
    sudo ln -s /usr/bin/clang-11 /usr/bin/clang
    sudo ln -s /usr/bin/clang++-11 /usr/bin/clang++
    sudo apt install libssl1.1 ocl-icd-opencl-dev fakeroot xorriso
    sudo aptitude install liblapack-dev
    

OpenCV from sources
===================

    git clone -b 4.5.0 https://github.com/opencv/opencv.git    
    cd opencv

    # To make opencv on M1 for x86 copy Terminal.app to Intel Terminal.app
    # open "Get Info", choose "Open using Rosetta"  
    # mkdir build_opencv_x86_64 && cd build_opencv_x86_64
    # cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/x86_64
    # ...
    
    mkdir build_opencv_arm64 && cd build_opencv_arm64
    cmake -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/arm64 \
    -DBUILD_SHARED_LIBS=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DWITH_EIGEN=ON\
    -DVIDEOIO_ENABLE_PLUGINS=OFF -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" ..
    # on centos add -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

    mkdir build_opencv_x86_64 && cd build_opencv_x86_64
    cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_OSX_DEPLOYMENT_TARGET=10.14 \
    -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/x86_64 \
    -DBUILD_SHARED_LIBS=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF -DVIDEOIO_ENABLE_PLUGINS=OFF -DWITH_EIGEN=ON\
    -DOPENCV_GENERATE_PKGCONFIG=ON DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
    -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" ..

    make -j7 && make install


    # Ubuntu 20.04
    sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev
    mkdir build_opencv_x86_64 && cd build_opencv_x86_64
    
    cmake -DWITH_CUDA=OFF -DWITH_OPENCL=ON -DWITH_OPENGL=ON -DWITH_V4L=ON -DBUILD_SHARED_LIBS=OFF -DWITH_EIGEN=ON\
    -DWITH_FFMPEG=ON -DVIDEOIO_ENABLE_PLUGINS=ON -DOPENCV_GENERATE_PKGCONFIG=ON -DBUILD_EXAMPLES=OFF\
    -DBUILD_TESTS=OFF -DBUILD_opencv_java=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" ..

Cuda
=======
Source: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=debnetwork

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo apt-get update
    sudo apt-get -y install cuda

    -   PATH includes /usr/local/cuda-11.1/bin
    -   LD_LIBRARY_PATH includes /usr/local/cuda-11.1/lib64, or, add /usr/local/cuda-11.1/lib64 to /etc/ld.so.conf and run ldconfig as root
    
    To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.1/bin
    ***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 455.00 is required for CUDA 11.1 functionality to work.
    To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

