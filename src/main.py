import sys
import os 
import re

sys.path.append(re.sub(r'/src/.*', '/src', os.path.abspath(__file__)))

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

from src.configs.config import *
from configs.constants import *

from models.image_segmentation_module import ImageSegmentationModule
from callbacks.save_results import SaveResults
from data.data_processor import DataProcessor
from utils import train, get_wabdb_logger

# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
PROJECT = "Finetuned"
NAME = "RESNET18_S2"

RESULTS_PATH = f"{PACKAGE_PATH}/results/{PROJECT}/{NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg: DataConfig = DATA_CONFIG_S2_C1
data_cfg.train_path = f"{DATA_PATH}/sentinel2/2021_seasons/Belgium_summer2021"

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
    

data_processor = DataProcessor(data_cfg)
module = ImageSegmentationModule(net_cfg)

logger = get_wabdb_logger(PROJECT, NAME, log_model='all')
logger.config["data"] = data_cfg.__dict__
logger.config.update()

train(module, data_processor,
      num_epochs=10, batch_size=16, num_workers=0,
      callbacks=[checkpoint_callback, stopping_callback],
      logger=logger)

wandb.finish()
