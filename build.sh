#!/bin/bash
mkdir build
cd build
opencv_dir=opencv-mobile-4.12.0-ubuntu-2404
opencv_url=https://github.com/nihui/opencv-mobile/releases/download/v34/$opencv_dir.zip

# Check if the folder exists
if [ ! -d "$opencv_dir" ]; then
    if [ ! -f "$opencv_dir.zip" ]; then
        # Download the file
        echo "Downloading $opencv_url"
        wget "$opencv_url" -O "$opencv_dir.zip"
    else
        echo "$opencv_dir.zip already exists"
    fi
    # Extract the file
    echo "Extracting unzip $opencv_dir.zip"
    unzip $opencv_dir.zip
else
    echo "$opencv_dir already exists"
fi

cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$PWD/$opencv_dir/lib/cmake/opencv4 ..
cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=$PWD/$opencv_dir/lib/cmake/opencv4 ..

make -j16
make install