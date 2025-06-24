#!/bin/bash
module purge
module restore
module load singularitypro/4.1.2
module load gpu/0.15.4
module load cuda12.2/toolkit/12.2.2

# rm -rf build    
# mkdir -p build

singularity exec --nv \
  -B "$(pwd)":/workspace \
  -B /scratch/yuehaicen:/scratch/yuehaicen:rw \
  cellgpu_deps_expanse.sif \
  bash -lc "\
    export PATH=/usr/local/cuda/bin:\$PATH; \
    cd /workspace/build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DCUDSS_ROOT=/opt/cudss && \
    make VERBOSE=1 -j\$(nproc)"