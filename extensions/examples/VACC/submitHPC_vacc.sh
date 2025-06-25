#!/bin/bash
#SBATCH --partition=nvgpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --time=0:20:00


module load gcc/13.3.0-xp3epyt
module load cuda/12.6.2
module load netcdf-cxx/4.3.1
module load cgal/5.6-ng5gssh

./voronoi_activeBD_friction.out 4000