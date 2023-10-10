import os
from copy import deepcopy
from configs.config import *

SECRET_KEY = os.environ.get("IN_DOCKER_CONTAINER", False)

if SECRET_KEY:
    PACKAGE_PATH = "/app"
    DATA_PATH = "/data"
else:
    # PACKAGE_PATH = "/home/k/kyselica12/work/forest_segmentation"
    # DATA_PATH = "/home/k/kyselica12/work/data"
    PACKAGE_PATH = '/home/daniel/Documents/work/forest_segmentation'
    DATA_PATH = '/media/daniel/data1'

ALL_BANDS_LIST = list(Sentinel2Bands)
RGB_BANDS_LIST = [Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2]
ALL_CLASSES_SET = set(ESAWorldCover)

WANDB_API_KEY = "b73e051ec86e9d3e56a2d2c47f1e3661a1b2a4db"

IGNORE_INDDEX = 255

DEFAULT = DataConfig(
    val_size=0.2,
    load=True,
    bands= RGB_BANDS_LIST, # RGB
    classes= set([ESAWorldCover.TREES]),
    train_path=f"{DATA_PATH}/sentinel2/2021_seasons/Belgium_summer2021",
    compute_mean_std=True,
    output_path=f"{PACKAGE_PATH}/resources/datasets"
)

DATA_CONFIG_RGB = deepcopy(DEFAULT)
DATA_CONFIG_RGB.bands = [Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2]

DATA_CONFIG_S2_A = deepcopy(DEFAULT)
DATA_CONFIG_S2_A.bands = ALL_BANDS_LIST

DATA_CONFIG_S2_C1 = deepcopy(DATA_CONFIG_S2_A)
DATA_CONFIG_S2_C1.use_level_C1 = True


DEFAULT_NET_CONFIG = NetConfig(
    architecture=NetworkArchitectures.UNET,
    args={
        'encoder_name': 'timm-mobilenetv3_small_minimal_100',
        'encoder_weights': 'imagenet',
    }
)

NET_CONFIG_IMAGENET_MOBILENET = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_IMAGENET_MOBILENET.args = {
    'encoder_name': 'timm-mobilenetv3_small_minimal_100',
    'encoder_weights': 'imagenet',
}

NET_CONFIG_IMAGENET_RESNET18 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_IMAGENET_RESNET18.args = {
    'encoder_name': 'resnet18',
    'encoder_weights': 'imagenet',
}

NET_CONFIG_IMAGENET_RESNET50 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_IMAGENET_RESNET50.args = {
    'encoder_name': 'resnet50',
    'encoder_weights': 'imagenet',
}

NET_CONFIG_S2_RESNET50 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_S2_RESNET50.args = {
    'encoder_name': CustomNets.RESNET50,
    'encoder_weights': CustomNetWeights.RESNET50,
}

NET_CONFIG_S2_RESNET18 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_S2_RESNET18.args = {
    'encoder_name': CustomNets.RESNET18,
    'encoder_weights': CustomNetWeights.RESNET18,
}