from poplib import CR
import sys
import os 
import re
from typing import Tuple
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

sys.path.append(re.sub(r'/src/.*', '/src', os.path.abspath(__file__)))

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

from configs.config import *
from configs.constants import *

from models.image_segmentation_module import ImageSegmentationModule
from callbacks.save_results import SaveResults
from data.data_processor import DataProcessor
from utils import train, get_wabdb_logger, register_SSL4EO_S12_encoders
from experiment import Experiment

# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
PROJECT = "Super-Resolution"
NAME = "HR->HR"
# NAME = "LR->HR"

RESULTS_PATH = f"{PACKAGE_PATH}/results/{PROJECT}/{NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg = DataConfig(
    train_path=f"{DATA_PATH}/sentinel2/dataset/HR_to_HR/docker",
    grid_path=f"{DATA_PATH}/sentinel2/dataset/HR_to_HR/docker/grid_features.json",
    output_path=f"{PACKAGE_PATH}/resources/datasets",
    classes= set([ESAWorldCover.TREES]),
    bands = [0,1,2],
    width=512,
    height=512,
    val_size=0.1,
    non_zero_ratio=0.5,
    first_id=1,
    stabilization_scale_factor=255,
    use_level_C1=False,
    improved_mask=False,
    compute_mean_std=False,
    load=True,
)


net_cfg: NetConfig = NET_CONFIG_IMAGENET_RESNET18
net_cfg.loss_cfg = CROSS_ENTROPY_LOSS

register_SSL4EO_S12_encoders(PACKAGE_PATH)  

torch.manual_seed(42)

dp = DataProcessor(data_cfg)
module = ImageSegmentationModule(**net_cfg.__dict__)
module.improved_mask = False
module.mask_difference_value = 1

chckpt_callback= ModelCheckpoint(
    dirpath=f'{PACKAGE_PATH}/resources/models/{PROJECT}/{NAME}',
    filename='{epoch}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    save_top_k=3,
)
os.makedirs(f'{PACKAGE_PATH}/resources/models/{PROJECT}/{NAME}', exist_ok=True)
stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
)

callbacks = [chckpt_callback, stopping_callback]
# callbacks = []

logger = get_wabdb_logger(PROJECT, NAME)
# logger = None

train(module, dp, 1000, 32, 0, callbacks, logger)
