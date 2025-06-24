#!/bin/bash
#SBATCH -A emu117
#SBATCH --partition=gpu-shared
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=0:10:00

module purge
module restore
module load singularitypro/4.1.2
module load gpu/0.15.4
module load cuda12.2/toolkit/12.2.2

singularity exec --nv -B $(pwd):/workspace ~/cellGPU_friction/cellGPU_adaptive/cellgpu_deps_expanse.sif /workspace/build/voronoi_activeBD_friction.out 4000
