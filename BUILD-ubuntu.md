# Requirements (Ubuntu 22.04)

    sudo apt-get update 
    sudo apt-get upgrade
    sudo apt-get install libc++-dev libc++abi-dev
    sudo apt-get install libomp-dev

    sudo apt-get install ocl-icd-opencl-dev fakeroot xorriso
    sudo apt install gfortran 

    # Lapack support
    git clone https://github.com/Reference-LAPACK/lapack-release.git
    cd lapack-release && mkdir build && cd build
    cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_COMPILER=clang  ..
    make -j6 && sudo make install

    # Cuda support
    
    # works with ubuntu22.04 drivers 
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    
    # works with new nvidia drivers only, > 545.xx.xx
    # wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda_12.3.1_545.23.08_linux.run
    
    chmod a+x cuda_12.3.1_545.23.08_linux.run
    sudo sh cuda_12.3.1_545.23.08_linux.run --no-drm --toolkit

    # OpenCV support
    sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
    git clone https://github.com/opencv/opencv.git
    cd opencv
    mkdir build_opencv_x86_64 && cd build_opencv_x86_64
    cmake -DWITH_CUDA=OFF -DWITH_OPENCL=ON -DWITH_OPENGL=ON -DWITH_V4L=ON -DBUILD_SHARED_LIBS=OFF -DWITH_EIGEN=OFF\
    -DWITH_FFMPEG=ON -DVIDEOIO_ENABLE_PLUGINS=ON -DOPENCV_GENERATE_PKGCONFIG=ON -DBUILD_EXAMPLES=OFF\
    -DBUILD_TESTS=OFF -DBUILD_opencv_java=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=".a" ..
    make -j12 && sudo make install

# Debug support
    export INSTALL_PREFIX=~/Develop/local/ubuntu/x86_64/cuda/dehancer
    export PKG_CONFIG_PATH=~/Develop/local/ubuntu/x86_64/dehancer/lib/pkgconfig
    cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=/usr/bin/make -DCMAKE_C_COMPILER=/usr/bin/clang\
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DBUILD_TESTING=ON -DPRINT_DEBUG=ON -DPRINT_KERNELS_DEBUG=OFF\
    -DDEHANCER_GPU_CUDA=ON -DDEHANCER_GPU_OPENCL=OFF -DDEHANCER_GPU_METAL=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=.a\
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -G "CodeBlocks - Unix Makefiles" ..     
    cmake --build . --config Debug -j8 && cmake --install . --config Debug

# Release support
    export INSTALL_PREFIX=~/Develop/local/release/ubuntu/x86_64/cuda/dehancer
    export PKG_CONFIG_PATH=~/Develop/local/release/ubuntu/x86_64/dehancer/lib/pkgconfig
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=/usr/bin/make -DCMAKE_C_COMPILER=/usr/bin/clang\
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DBUILD_TESTING=OFF -DPRINT_DEBUG=OFF -DPRINT_KERNELS_DEBUG=OFF\
    -DDEHANCER_GPU_CUDA=ON -DDEHANCER_GPU_OPENCL=OFF -DDEHANCER_GPU_METAL=OFF -DCMAKE_FIND_LIBRARY_SUFFIXES=.a\
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -G "CodeBlocks - Unix Makefiles" ..     
    cmake --build . --config Release -j8 && cmake --install . --config Release