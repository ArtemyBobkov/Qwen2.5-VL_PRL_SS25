#!/bin/bash

# Distributed training configuration

# Modified world size because else lead to CUDA error: invalid device id
WORLD_SIZE=1
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:1}
NPROC_PER_NODE=1

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=2e-7
batch_size=8
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=spatial_qa_univlg

# Output configuration
run_name="qwen2_5vl-univlg-precomputed-emb-finetune-small-mlp"
output_dir=/lustre/scratch/data/s94falmu_hpc-PLRSpatial/qwen_univlg_precomputed_emb_finetune_small_mlp

# nvidia-smi > ./nvidia-smi.log
# nvcc --version > ./nvcc-version.log

# export PYTHONPATH=$PYTHONPATH:$(realpath /home/s67abobk_hpc/univlg)
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages/:$PYTHONPATH"

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 147456 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --run_name ${run_name} \
    --report_to wandb \
    --precomputed_embeddings True \
    --precomputed_embedding_dim 256 \
    --precomputed_embeddings_path /lustre/scratch/data/s94falmu_hpc-PLRSpatial/univlg_embeddings
    "

# Launch training
~/.local/bin/uv run /home/s67abobk_hpc/univlg/.venv/bin/python3 -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
