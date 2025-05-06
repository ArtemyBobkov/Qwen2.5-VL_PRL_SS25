#!/bin/bash

#SBATCH --partition=mlgpu_devel

#SBATCH --time=0:30:00

#SBATCH --gpus=1

#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load Miniforge3
module load CUDA/12.2.0
source ~/.bashrc

apptainer build --force /lustre/scratch/data/s94falmu_hpc-PLRSpatial/cuda_qwenvlm.sif qwenvl_2_5.def
