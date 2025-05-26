#!/bin/bash

#SBATCH --partition=sgpu_devel

#SBATCH --time=00:45:00

#SBATCH --gpus=2
#SBATCH --ntasks=48

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
    --env WANDB_API_KEY= \
    --env SSL_CERT_FILE=/home/s67abobk_hpc/Qwen2.5-VL_PRL_SS25/cacert_wandb.pem \
    /lustre/scratch/data/s94falmu_hpc-PLRSpatial/cuda_qwenvlm.sif \
    ./scripts/finetune_spatialbot.sh

