#!/bin/bash

## Run this script the first time you build Chestnut from the top-level
## directory (the one with the top-level CMakeLists.txt file)

mkdir build
cd build
cmake .. || exit 1
make || exit 2
cd parser
ln -s ../../parser/chestnut.in .
./chestnut_parse < chestnut.in || exit 3

# link in chestnut output file
cd ../../chestnut/
ln -s ../build/parser/chestnut.cu .

# re-cmake things again
cd ../build
cmake -DBUILD_CHESTNUT=true .. || exit 4
make || exit 5
