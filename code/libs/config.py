import yaml

# DEFAULT defines the default params used for training / inference
# the parameters here will be overwritten if a yaml config is specified
DEFAULTS = {
    # default: single gpu
    "devices": ["cuda:0"],
    "dtype": "fp16",
    "model_name": "FCOS",
    # output folder that stores all log files and checkpoints
    "output_folder": None,
    "dataset": {
        "name": "MovieLens20M",
        # training / testing splits
        "train": "train",
        "test": "test",
        # folders that store image files / json annotations (following COCO format)
        "img_folder": None,
        "ann_folder": None,
    },
    "loader": {
        "batch_size": 4,
        "num_workers": 4,
    },
    "input": {
        # mean / std for input normalization
        # "img_mean": [0.485, 0.456, 0.406],
        # "img_std": [0.229, 0.224, 0.225],
    },
    # network architecture
    "model": {
        # type of backbone network
        "backbone": "resnet18",
        # if to freeze all batchnorms in the backbone
        # this is needed when training with a small batch size
        "backbone_freeze_bn": True,
        # output feature dimensions (assuming resnet18/34)
        "backbone_out_feats_dims": 512,
        # number of genres in the dataset
        "num_genres": 20,
    },
    "train_cfg": {
    },
    "test_cfg": {
        # threshold for positive class
        "score_thresh": 0.1,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "SGD",
        # solver params
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "learning_rate": 5e-3,
        # excluding the warmup epochs
        "epochs": 10,
        # if to use linear warmup
        "warmup": True,
        "warmup_epochs": 1,
        # lr scheduler: cosine / multistep
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    },
}


def _merge(src, dst):
    """
    :param src: source dictionary
    :param dst: target dictionary
    :return: None, dst is updated so that it uses defaults from src.
    """
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_default_config():
    config = DEFAULTS
    return config


def _update_config(config):
    config["model"].update(config["input"])
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config
