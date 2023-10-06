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
PROJECT = "test"
NAME = "RESNET18_S2"

RESULTS_PATH = f"{PACKAGE_PATH}/results/{PROJECT}/{NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg: DataConfig = DATA_CONFIG_S2_C1
data_cfg.train_path = f"{DATA_PATH}/sentinel2/2021_seasons/Belgium_summer2021"
data_cfg.compute_mean_std = False

net_cfg: NetConfig = NET_CONFIG_S2_RESNET18

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=f'{PACKAGE_PATH}/resources/models',
    filename=f'{PROJECT}-{NAME}-' + '{epoch:02d}-{val_loss:.2f}',
    save_top_k=3
)

stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min',
)
    
register_SSL4EO_S12_encoders(PACKAGE_PATH)   

# data_processor = DataProcessor(data_cfg)
# module = ImageSegmentationModule(net_cfg)

# logger = get_wabdb_logger(PROJECT, NAME, log_model='all')

# train(module, data_processor,
#       num_epochs=10, batch_size=5, num_workers=0,
#       callbacks=[checkpoint_callback, stopping_callback],
#       logger=logger)

class TestExperiment(Experiment):
    
    def process_option(self, desc, val) -> Tuple[DataConfig, NetConfig]:
        data_cfg, net_cfg =  super().process_option(desc, val)

        return data_cfg, net_cfg


options = [
    ("1", 1)
]

experiment = TestExperiment(PROJECT, data_cfg, net_cfg, log_to_wandb=True)
experiment.run(options, 1, 5, 0, [])