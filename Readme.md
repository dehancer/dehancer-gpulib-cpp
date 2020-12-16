Requirements
===========
    Nasm:
    brew install nasm
    or
    yum install nasm
    
LibJPEG Turbo
=============

    git clone https://github.com/libjpeg-turbo/libjpeg-turbo
    cd libjpeg-turbo && mkdir build && cd build
    cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFI  X=/usr/local ..
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
    mkdir build_opencv && cd build_opencv
    cmake -DBUILD_SHARED_LIBS=OFF -DWITH_FFMPEG=OFF -DWITH_V4L=OFF \
    -DVIDEOIO_ENABLE_PLUGINS=OFF -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" ..
    # on centos add -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    make -j7 && make install

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

