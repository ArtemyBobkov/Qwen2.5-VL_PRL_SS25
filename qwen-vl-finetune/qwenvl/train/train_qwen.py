# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

from transformers.activations import ACT2FN
from copy import deepcopy

import deepspeed
from dataclasses import dataclass

import torch.nn as nn

import sys
print("Python search paths:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")    

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


import numpy as np
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.train.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMLP
)

from qwenvl.train.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwenvl.data.data_qwen import make_supervised_data_module

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None

@dataclass
class Qwen2_5_VL_ModifiedVLMLPConfig:
    hidden_size: int = 256
    intermediate_size: int = 2048
    hidden_act="silu"
    output_size: int = 2048
    
class Qwen2_5_VL_ModifiedVLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.output_size = config.output_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.output_size, bias=bias)
        
        torch.nn.init.xavier_uniform_(self.gate_proj.weight)
        torch.nn.init.xavier_uniform_(self.up_proj.weight)
        torch.nn.init.xavier_uniform_(self.down_proj.weight)
        
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2_5_VL_UNIVLG_PrecomputedEmbeddingProjection(nn.Module):
    def __init__(self, hidden_size, output_size, bias: bool = False):
        super().__init__()
        config = Qwen2_5_VL_ModifiedVLMLPConfig(hidden_size=hidden_size)
        
        input_config = deepcopy(config)
        input_config.output_size = config.intermediate_size
        
        # intermediate_config = deepcopy(config)
        # intermediate_config.hidden_size = config.intermediate_size
        # intermediate_config.output_size = config.intermediate_size
        
        output_config = deepcopy(config)
        output_config.hidden_size = config.intermediate_size
        output_config.output_size = output_size
        
        self.mlp1 = Qwen2_5_VL_ModifiedVLMLP(input_config, bias=bias)
        # self.mlp2 = Qwen2_5_VL_ModifiedVLMLP(intermediate_config, bias=bias)
        # self.mlp2 = Qwen2_5_VL_ModifiedVLMLP(output_config, bias=bias)
        
        # self.precomputed_embedding_projection = nn.Linear(config.intermediate_size, config.vision_config.out_hidden_size, bias=False)
        
        print("MLP1", self.mlp1)
        # print("MLP2", self.mlp2)
        # print("MLP3", self.mlp3)
        
    def forward(self, precomputed_image_embeds):
        precomputed_image_embeds = precomputed_image_embeds.requires_grad_(True)
        image_embeds = self.mlp1(precomputed_image_embeds)
        # image_embeds = self.mlp2(image_embeds)
        # image_embeds = self.mlp3(image_embeds)
        return image_embeds

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        print("Tuning vision")
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        print("Tuning MLP")
        # for n, p in model.visual.merger.named_parameters():
        #     p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        print("Tuning LLM")
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
    
    # Handle precomputed embedding projection layer
    if hasattr(model, 'precomputed_embedding_projection') and model.precomputed_embedding_projection is not None:
        if model_args.tune_mm_mlp:  # Use the same flag as MLP tuning
            for p in model.precomputed_embedding_projection.parameters():
                p.requires_grad = True
            rank0_print("Precomputed embedding projection layer is trainable, trainable parameters: ", sum(p.numel() for p in model.precomputed_embedding_projection.parameters() if p.requires_grad))
        else:
            for p in model.precomputed_embedding_projection.parameters():
                p.requires_grad = False
            rank0_print("Precomputed embedding projection layer is frozen, trainable parameters: ", sum(p.numel() for p in model.precomputed_embedding_projection.parameters() if p.requires_grad))


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if not "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2_5_VLProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        #data_args.image_processor = AutoProcessor.from_pretrained(
        #    model_args.model_name_or_path,
        #).image_processor
        data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    # Configure precomputed embeddings if specified
    if model_args.precomputed_embedding_dim is not None:
        # Set the precomputed embedding dimension in the config
        model.config.precomputed_embedding_dim = model_args.precomputed_embedding_dim
        
        # The model already has a precomputed_embedding_projection layer built-in
        # We just need to ensure it's properly configured
        if hasattr(model, 'precomputed_embedding_projection') and model.precomputed_embedding_projection is not None:
            # Update the projection layer if the input dimension doesn't match
            if model.precomputed_embedding_projection.in_features != model_args.precomputed_embedding_dim:
                model.precomputed_embedding_projection = Qwen2_5_VL_UNIVLG_PrecomputedEmbeddingProjection(
                    model_args.precomputed_embedding_dim, 
                    model.config.vision_config.out_hidden_size,
                )
                rank0_print(f"Updated precomputed embedding projection: {model_args.precomputed_embedding_dim} -> {model.config.vision_config.out_hidden_size}")
        else:
            model.precomputed_embedding_projection = Qwen2_5_VL_UNIVLG_PrecomputedEmbeddingProjection(
                model_args.precomputed_embedding_dim, 
                model.config.vision_config.out_hidden_size,
            )
            rank0_print(f"Created precomputed embedding projection: {model_args.precomputed_embedding_dim} -> {model.config.vision_config.out_hidden_size}")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # # Add gradient debugging hooks after set_model function
    # if training_args.local_rank == 0:
    #     def make_hook(n):
    #         def hook(m, grad_input, grad_output):
    #             if len(grad_input) > 0:
    #                 gi0 = grad_input[0]
    #                 go0 = grad_output[0]
    #                 print(f"{m}.{n}: grad_input norm = {gi0.norm() if gi0 is not None else None}; "
    #                     f"grad_output norm = {go0.norm() if go0 is not None else None}")
    #             elif len(grad_output) > 0:
    #                 go0 = grad_output[0]
    #                 print(f"{m}.{n}: no grad_input")
    #                 print(f"{m}.{n}: grad_output norm = {go0.norm() if go0 is not None else None}")
    #             else:
    #                 print(f"{m}.{n}: no grad_input")
    #                 print(f"{m}.{n}: no grad_output")
                    
        
    #         return hook
        
    # print(model)

    params = list(model.parameters())   
    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        if torch.distributed.get_rank() == 0:
            # for n, p in zip(model.state_dict().keys(), params):
                # print(n, p.shape, p.requires_grad)
            print("trainable =", sum(p.numel() for p in params if p.requires_grad))

    if torch.distributed.get_rank() == 0:
        params = list(model.parameters())
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join("/home/s67abobk_hpc/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3/", "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
