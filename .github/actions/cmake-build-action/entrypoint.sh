#!/bin/sh -l

echo "::group::Installing Open MPI"
apt-get install -y libopenmpi-dev
echo "::endgroup::"

export CC=gcc
export CXX=g++
export build_mode=Release
cmake -B build -DCMAKE_BUILD_TYPE=$build_mode
cmake --build build/
