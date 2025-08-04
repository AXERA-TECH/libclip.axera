#!/bin/bash

echo "aarch64"

mkdir build_aarch64
cd build_aarch64

opencv_aarch64_url=https://github.com/ZHEQIUSHUI/assets/releases/download/ax650/libopencv-4.5.5-aarch64.zip


# Check if the folder exists
if [ ! -d "libopencv-4.5.5-aarch64" ]; then
    if [ ! -f "libopencv-4.5.5-aarch64.zip" ]; then
        # Download the file
        echo "Downloading $opencv_aarch64_url"
        wget "$opencv_aarch64_url" -O "libopencv-4.5.5-aarch64.zip"
    else
        echo "libopencv-4.5.5-aarch64.zip already exists"
    fi
    # Extract the file
    echo "Extracting unzip libopencv-4.5.5-aarch64.zip"
    unzip libopencv-4.5.5-aarch64.zip
else
    echo "libopencv-4.5.5-aarch64 already exists"
fi


# 下载失败可以使用其他方式下载并放到在 $build_dir 目录，参考如下命令解压
URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

aarch64-none-linux-gnu-gcc -v
if [ $? -ne 0 ]; then
    # Check if the file exists
    if [ ! -f "$FOLDER.tar.xz" ]; then
        # Download the file
        echo "Downloading $URL"
        wget "$URL" -O "$FOLDER.tar.xz"
    else
        echo "$FOLDER.tar.xz already exists"
    fi

    # Check if the folder exists
    if [ ! -d "$FOLDER" ]; then
        # Extract the file
        echo "Extracting $FOLDER.tar.xz"
        tar -xf "$FOLDER.tar.xz"
    else
        echo "$FOLDER already exists"
    fi

    export PATH=$PATH:$PWD/$FOLDER/bin/
    aarch64-none-linux-gnu-gcc -v
    if [ $? -ne 0 ]; then
        echo "Error: aarch64-none-linux-gnu-gcc not found"
        exit 1
    fi
else
    echo "aarch64-none-linux-gnu-gcc already exists"
fi


cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
-DOpenCV_DIR=$PWD/libopencv-4.5.5-aarch64/lib/cmake/opencv4 \
-DCMAKE_BUILD_TYPE=Release \
..

cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
-DOpenCV_DIR=$PWD/libopencv-4.5.5-aarch64/lib/cmake/opencv4 \
-DCMAKE_BUILD_TYPE=Release \
..


make -j16
make install