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
    cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
    make -j4 && make install 

Centos7 (based for DaVinci Resolve 16) 
============
    yum install llvm-toolset-7
    scl enable llvm-toolset-7 bash
    cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_TESTING=ON ..
    
OpenCV from sources
===================

    git clone -b 4.5.0 https://github.com/opencv/opencv.git    
    cd opencv
    mkdir build_opencv && cd build_opencv
    cmake -DBUILD_SHARED_LIBS=OFF -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" ..
    # on centos add -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    make -j7 && make install