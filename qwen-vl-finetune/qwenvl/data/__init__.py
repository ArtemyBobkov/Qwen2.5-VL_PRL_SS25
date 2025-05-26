import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

BUNNY_V1_0_PRETRAIN = {
    "annotation_path": "/lustre/scratch/data/s94falmu_hpc-PLRSpatial/Bunny-v1_0-data/pretrain/bunny_pretrain_laion_2m.json",
    "data_path": "/lustre/scratch/data/s94falmu_hpc-PLRSpatial/Bunny-v1_0-data/pretrain/images",
}

BUNNY_V1_0_FINETUNE = {
    "annotation_path": "/lustre/scratch/data/s94falmu_hpc-PLRSpatial/Bunny-v1_0-data/finetune/bunny_695k.json",
    "data_path": "/lustre/scratch/data/s94falmu_hpc-PLRSpatial/Bunny-v1_0-data/finetune/images",
}

SPATIAL_QA = {
    "annotation_path": "/lustre/scratch/data/s94falmu_hpc-PLRSpatial/SpatialQA/SpatialQA.json",
    "data_path": "/lustre/scratch/data/s94falmu_hpc-PLRSpatial/Bunny-v1_0-data/finetune/images",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "bunny_v1_0_pretrain": BUNNY_V1_0_PRETRAIN,
    "bunny_v1_0_pretrain%10": BUNNY_V1_0_PRETRAIN,
    "bunny_v1_0_finetune": BUNNY_V1_0_FINETUNE,
    "bunny_v1_0_finetune%10": BUNNY_V1_0_FINETUNE,
    "spatial_qa": SPATIAL_QA,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["spatial_qa"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
