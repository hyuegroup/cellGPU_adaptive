Bootstrap: docker
From: nvidia/cuda:12.1.1-devel-ubuntu20.04

%labels
    Author Haicen YUe
    Version v0.1

%environment
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda
    export CUDSS_ROOT=/opt/cudss
    export LD_LIBRARY_PATH=$CUDSS_ROOT/lib64:$LD_LIBRARY_PATH

%post
    export DEBIAN_FRONTEND=noninteractive
    export TZ=America/New_York

     # build CMake ≥3.22 from source
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential wget ca-certificates \
    && wget https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0.tar.gz \
    && tar xzf cmake-3.24.0.tar.gz \
    && cd cmake-3.24.0 \
    && ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF \
    && make -j$(nproc) && make install \
    && cd .. && rm -rf cmake-3.24.0* \

    apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget ca-certificates \
        libgmp-dev libmpfr-dev libboost-all-dev \
        libeigen3-dev libhdf5-dev hdf5-tools libcgal-dev\
        libnuma-dev libtbb-dev libopenmpi-dev \
        tzdata \
      && rm -rf /var/lib/apt/lists/*

     # --- Install the cuDSS preview into /opt/cudss ---
    # (grab the right version for CUDA 12.1 or your target)
    mkdir -p /opt/cudss
    wget -qO cudss.tar.xz \
      https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.6.0.5_cuda12-archive.tar.xz
    tar -xJf cudss.tar.xz -C /opt/cudss --strip-components=1
    rm cudss.tar.xz

%runscript
    exec /opt/cellGPU_adaptive/build/voronoi_activeBD_friction.out "$@"

%help
    Run your cellGPU_adaptive executable inside a CUDA-enabled container:
      singularity run --nv cellgpu.sif 4000
