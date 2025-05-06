#!/bin/bash

#SBATCH --partition=sgpu_long

#SBATCH --time=36:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=32

#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err


# prep
module load Miniforge3
module load CUDA/12.2.0
source ~/.bashrc
# conda activate plr

# run
# nvidia-smi > /home/s67abobk_hpc/nvidia-smi.log
# nvcc --version > /home/s67abobk_hpc/nvcc-version.log
apptainer exec \
    --nv \
    --bind /lustre/scratch/data/s94falmu_hpc-PLRSpatial \
    --env WANDB_API_KEY=%wandb_api_key% \
    --env VLLM_USE_V1=1 \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    /lustre/scratch/data/s94falmu_hpc-PLRSpatial/cuda_qwenvlm.sif \
    ./scripts/pretrain_bunny.sh

