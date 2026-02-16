#!/bin/bash

# Build script for OpenCLWrapper library

cd "$(dirname "$0")"

mkdir -p build
cd build
cmake ..
make -j$(nproc)

