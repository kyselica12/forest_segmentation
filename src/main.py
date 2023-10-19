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
PROJECT = "Test"
NAME = "<PLACEHOLDER>"

RESULTS_PATH = f"{PACKAGE_PATH}/results/{PROJECT}/{NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg: DataConfig = DATA_CONFIG_S2_C1
data_cfg.train_path = f"{DATA_PATH}/sentinel2/2021_seasons"
data_cfg.val_size = 0.1
data_cfg.compute_mean_std = True
data_cfg.non_zero_ratio = 0.5

net_cfg: NetConfig = NET_CONFIG_S2_RESNET18

register_SSL4EO_S12_encoders(PACKAGE_PATH)   


module = ImageSegmentationModule(**net_cfg.__dict__).cuda()
dp = DataProcessor(data_cfg)

callbacks = []

logger = get_wabdb_logger(PROJECT, NAME, log_model='all')

train(module, dp, 1000, 16, 0, callbacks, logger)
