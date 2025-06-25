#!/bin/bash
module load gcc/13.3.0-xp3epyt
module load cuda/12.6.2
module load netcdf-cxx/4.3.1
module load cgal/5.6-ng5gssh

CUDSS_ROOT="${HOME}/cudss"
BUILD_TYPE=Release

export CPATH="${CUDSS_ROOT}/include:${CPATH:-}"
export LD_LIBRARY_PATH="${CUDSS_ROOT}/lib:${LD_LIBRARY_PATH:-}"

cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCUDSS_INCLUDE_DIR="${CUDSS_ROOT}/include" \
  -DCUDSS_LIB="${CUDSS_ROOT}/lib/libcudss.so" \
  -DCMAKE_CUDA_COMPILER="$(which nvcc)"

make
