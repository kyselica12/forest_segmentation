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
PROJECT = "Improved masks"
NAME = "Value"

IMPROVED_MASK = True
DIFFERENCE_IMPORTANCE = 2
NAME += f"_{DIFFERENCE_IMPORTANCE}"


RESULTS_PATH = f"{PACKAGE_PATH}/results/{PROJECT}/{NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg: DataConfig = DATA_CONFIG_RGB
data_cfg.train_path = f"{DATA_PATH}/sentinel2/2021_seasons/Belgium_summer2021"
data_cfg.val_size = 0.1
data_cfg.compute_mean_std = True
data_cfg.non_zero_ratio = 0.5
data_cfg.improved_mask = IMPROVED_MASK

net_cfg: NetConfig = NET_CONFIG_IMAGENET_MOBILENET

register_SSL4EO_S12_encoders(PACKAGE_PATH)   

torch.manual_seed(42)
module = ImageSegmentationModule(**net_cfg.__dict__).cuda()
module.improved_mask = IMPROVED_MASK
module.mask_difference_value = DIFFERENCE_IMPORTANCE

dp = DataProcessor(data_cfg)

callbacks = []
 
logger = get_wabdb_logger(PROJECT, NAME, log_model='all')
# logger = None

train(module, dp, 10, 10, 0, callbacks, logger)
