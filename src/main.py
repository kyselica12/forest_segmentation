import sys
import os 
import re
from typing import Tuple

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
PROJECT = "Finetuned_2"
NAME = "<PLACEHOLDER>"

RESULTS_PATH = f"{PACKAGE_PATH}/results/{PROJECT}/{NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg: DataConfig = DATA_CONFIG_S2_C1
data_cfg.train_path = f"{DATA_PATH}/2021_seasons"
data_cfg.val_size = 0.1
data_cfg.compute_mean_std = True

net_cfg: NetConfig = None  

register_SSL4EO_S12_encoders(PACKAGE_PATH)   

options = [
    ('Resnet18_S2', NET_CONFIG_S2_RESNET18),
    ('Resnet50_S2', NET_CONFIG_S2_RESNET50),
    ('Resnet18_IMAGENET', NET_CONFIG_IMAGENET_RESNET18),
    ('Resnet50_IMAGENET', NET_CONFIG_IMAGENET_RESNET50)
]

class FinetuneExperiment(Experiment):
    
    def process_option(self, desc, val) -> Tuple[DataConfig, NetConfig]:
        data_cfg, _ = super().process_option(desc, val)

        net_cfg = val
        
        if 'IMAGENET' in desc:
            data_cfg.use_level_C1 = False
            data_cfg.bands = RGB_BANDS_LIST

        return data_cfg, net_cfg
    
    def get_callbacks(self, desc, val):
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath=f'{PACKAGE_PATH}/resources/models',
            filename=f'{PROJECT}-{desc}-' + '{epoch:02d}-{val_loss:.2f}',
            save_top_k=3
        )

        stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
        )
        
        return [checkpoint_callback, stopping_callback]
        

experiment = FinetuneExperiment(PROJECT, data_cfg, net_cfg, log_to_wandb=True)

experiment.run(options,
               n_epochs=1000,
               batch_size=10,
               num_workers=0,
               )
