import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from pathlib import Path

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
    precomputed_embeddings: bool = False,
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            # I suppose there are issues with image_thw
            # It wants to be padded depending on temporal dimension, but number of images does not match that
            # Hopefully small manual change will be enough
            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    
                    # For precomputed embeddings, keep only the first visual tag
                    max_tags = 1 if precomputed_embeddings else len(parts) - 1
                    
                    for i in range(min(max_tags, len(parts) - 1)):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    
                    # For precomputed embeddings, join all remaining parts without visual tags
                    if precomputed_embeddings and len(parts) > 2:
                        new_parts.append("".join(parts[1:]))
                    else:
                        new_parts.append(parts[-1])
                        
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        # random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
        self.precomputed_embeddings = getattr(data_args, "precomputed_embeddings", False)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file)
        
        channels = len(image.getbands())
        if channels == 1:
            img = np.array(image, dtype=np.int32)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.int32)
            three_channel_array[:, :, 0] = (img // 1024) * 4
            three_channel_array[:, :, 1] = (img // 32) * 8
            three_channel_array[:, :, 2] = (img % 32) * 8
            three_channel_array = three_channel_array.astype(np.uint8)
            image = Image.fromarray(three_channel_array, 'RGB')
        else:
            image = image.convert("RGB")
        
        # Apply square padding if configured (same as data_utils.py)
        if getattr(self.data_args, 'image_aspect_ratio', None) == 'pad':
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        
        # Additional image processing modes from Conversation class
        image_process_mode = getattr(self.data_args, 'image_process_mode', None)
        if image_process_mode == "Pad":
            image = expand2square(image, (122, 116, 104))
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        elif image_process_mode in ["Default", "Crop"]:
            pass
        # If no image_process_mode specified, keep existing behavior
        
        # Aspect ratio processing from Conversation class (applied after other processing)
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 800, 400
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if longest_edge != max(image.size):
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def process_precomputed_embeddings(self, precomputed_path):
        """
        Process precomputed embeddings from a file
        
        Args:
            precomputed_path: Path to the precomputed embeddings file (.pt or .pth)
            
        Returns:
            precomputed_embeddings: Tensor of shape [num_tokens, embedding_dim]
            grid_thw: Tensor of shape [3] containing temporal, height, width dimensions
        """
        # Load precomputed embeddings
        if precomputed_path.endswith('.pt') or precomputed_path.endswith('.pth'):
            embedding_data = torch.load(
                precomputed_path, 
                map_location="cpu", 
                weights_only=False,
            )
            
            # Expected format: {'embeddings': tensor, 'grid_thw': tensor}
            if isinstance(embedding_data, dict):
                precomputed_embeddings = embedding_data['embeddings']
                precomputed_embeddings = precomputed_embeddings.to(torch.float32)
                grid_thw = embedding_data['grid_thw']
                grid_thw = torch.tensor([precomputed_embeddings.shape[0], 1, 1])
            else:
                # If it's just a tensor, assume it's the embeddings
                precomputed_embeddings = embedding_data
                precomputed_embeddings = precomputed_embeddings.reshape(-1, 256)
                precomputed_embeddings = precomputed_embeddings.to(torch.float32)
                # Calculate grid_thw from embedding shape
                num_tokens = precomputed_embeddings.shape[0]
                # Assuming square spatial layout and temporal dim = 1
                sqrt_tokens = int(math.sqrt(num_tokens))
                grid_thw = torch.tensor([num_tokens, 1, 1])
        else:
            raise ValueError(f"Unsupported precomputed embedding format: {precomputed_path}")
        
        # added detach to make sure they are without gradients
        precomputed_embeddings = precomputed_embeddings.detach()
        
        return precomputed_embeddings, grid_thw

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e, e.__traceback__)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        video = None
        # print(f"sources: {sources}")
        if self.precomputed_embeddings and ("image" in sources[0] or "video" in sources[0]):
            # print("GOT INTO PRECOMPUTED EMBEDDINGS")
            # Handle precomputed embeddings
            univlg_embeddings_folder = self.data_args.precomputed_embeddings_path
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            # image_subfolder = Path(image_file).parent
            # image_file_name = Path(image_file).stem
            # precomputed_embedding_path = os.path.join(UNIVLG_EMBEDDING_FOLDER, f"{str(image_subfolder)}_univlg_encoder_output", image_file_name)
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    for file in image_file:
                        image_subfolder = Path(file).parent
                        if "_d" in image_subfolder.name:
                            image_subfolder = str(image_subfolder).replace("_d", "")
                        image_file_name = Path(file).stem
                        precomputed_embedding_path = os.path.join(univlg_embeddings_folder, f"{str(image_subfolder)}_univlg_encoder_output", f"{image_file_name}.pt")
                        # print("PRECOMPUTED EMBEDDING PATH", precomputed_embedding_path)
                        results = [self.process_precomputed_embeddings(precomputed_embedding_path)]
                        # print(f"precomputed_embedding_path: {precomputed_embedding_path}")
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_subfolder = Path(image_file).parent
                    image_file_name = Path(image_file).stem
                    precomputed_embedding_path = os.path.join(univlg_embeddings_folder, f"{str(image_subfolder)}_univlg_encoder_output", f"{image_file_name}.pt")
                    # print("PRECOMPUTED EMBEDDING PATH", precomputed_embedding_path)
                    image, grid_thw = self.process_precomputed_embeddings(precomputed_embedding_path)
                    image = [image]
            else:
                # precomputed_file = os.path.join(image_folder, precomputed_file)
                image_subfolder = Path(image_file).parent
                image_file_name = Path(image_file).stem
                precomputed_embedding_path = os.path.join(univlg_embeddings_folder, f"{str(image_subfolder)}_univlg_encoder_output", f"{image_file_name}.pt")
                image, grid_thw = self.process_precomputed_embeddings(precomputed_embedding_path)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod()
                for merged_thw in grid_thw_merged
            ]
            # print("grid_thw_merged", grid_thw_merged, grid_thw)
            sources = copy.deepcopy([e["conversations"] for e in sources])
            # print("start preprocess_qwen_2_visual")
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image", 
                precomputed_embeddings=self.precomputed_embeddings
            )
            # print("end preprocess_qwen_2_visual")
            position_ids, _ = self.get_rope_index(
                1, # self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
            # print(f"Precomputed embeddings in lazy dataset image")
            # print(data_dict["input_ids"].shape, data_dict["input_ids"])
            image_token_mask = data_dict["input_ids"] == IMAGE_TOKEN_INDEX
            image_token_mask = torch.repeat_interleave(image_token_mask, 3, dim=0).unsqueeze(1)
            # print(position_ids.shape, position_ids)
            position_ids[image_token_mask] = 0
        elif "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image",
                precomputed_embeddings=self.precomputed_embeddings
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
            if self.precomputed_embeddings:
                print(f"Precomputed embeddings in lazy dataset, image part")
                # Find and zero out image token positions
                image_token_mask = data_dict["input_ids"] == IMAGE_TOKEN_INDEX
                image_token_mask = torch.repeat_interleave(image_token_mask, 3, dim=0).unsqueeze(1)
                position_ids[image_token_mask] = 0
                # real_grid_thw = grid_thw
        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video",
                precomputed_embeddings=self.precomputed_embeddings
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
            if self.precomputed_embeddings:
                # print(f"Precomputed embeddings in lazy dataset video")
                # Find and zero out image token positions
                image_token_mask = data_dict["input_ids"] == IMAGE_TOKEN_INDEX
                image_token_mask = torch.repeat_interleave(image_token_mask, 3, dim=0).unsqueeze(1)
                position_ids[image_token_mask] = 0
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged,
                precomputed_embeddings=self.precomputed_embeddings
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )

        if self.precomputed_embeddings and ("image" in self.list_data_dict[i] or "video" in self.list_data_dict[i]):
            data_dict["pixel_values"] = image
            data_dict["precomputed_image_embeds"] = image
            data_dict["image_grid_thw"] = grid_thw
            # data_dict["real_grid_thw"] = real_grid_thw
        elif "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw
        # print("PRECOMPUTED EMBEDS IN DATA DICT", data_dict)

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        # Handle precomputed embeddings
        precomputed_embeddings = list(
            itertools.chain(
                *(
                    instance["precomputed_image_embeds"]
                    for instance in instances
                    if "precomputed_image_embeds" in instance
                )
            )
        )
        # print("INSTANCES", instances)
        # print("PRECOMPUTED EMBEDS", [emb.shape for emb in precomputed_embeddings])
        if len(precomputed_embeddings) != 0:
            
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
            
            concat_precomputed = torch.cat([emb for emb in precomputed_embeddings], dim=0)
            assert grid_thw.sum(dim=0)[0] == concat_precomputed.shape[0], "grid_thw and precomputed embeddings have different number of tokens: {} != {}".format(grid_thw.sum(dim=0)[0], concat_precomputed.shape[0])
        else:
            concat_precomputed = None
            

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["precomputed_image_embeds"] = concat_precomputed
        batch["position_ids"] = position_ids
        
        print("BATCH IMAGE EMBEDS", batch["precomputed_image_embeds"].shape)
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor(
            [0] + [len(seq) for seq in input_ids], dtype=torch.int32
        )
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        # print("INSTANCES", instances)

        # Handle precomputed embeddings
        precomputed_embeddings = list(
            itertools.chain(
                *(
                    instance["precomputed_image_embeds"]
                    for instance in instances
                    if "precomputed_image_embeds" in instance
                )
            )
        )
        # print("INSTANCES", instances)
        # print("PRECOMPUTED EMBEDS", [emb.shape for emb in precomputed_embeddings])
        if len(precomputed_embeddings) != 0:
            
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
            
            concat_precomputed = torch.cat([emb for emb in precomputed_embeddings], dim=0)
            assert grid_thw.sum(dim=0)[0] == concat_precomputed.shape[0], "grid_thw and precomputed embeddings have different number of tokens: {} != {}".format(grid_thw.sum(dim=0)[0], concat_precomputed.shape[0])
        else:
            concat_precomputed = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["precomputed_image_embeds"] = concat_precomputed
        # print("BATCH IMAGE EMBEDS", batch["precomputed_image_embeds"].shape)
        # batch["real_grid_thw"] = real_grid_thw

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
