import sys
import os 
import re

sys.path.append(re.sub(r'/src/.*', '/src', os.path.abspath(__file__)))

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config import *
from configs.network_configs import *
from configs.data_configs import *

from experiment import Experiment
from callbacks.save_results import SaveResults
from data.data_processor import DataProcessor
from utils import train

# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
PROJECT = "Finetuned"
NAME = "RESNET18_S2"

RESULTS_PATH = f"{PACKAGE_PATH}/results/{PROJECT}/{NAME}"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg: DataConfig = DATA_CONFIG_S2_C1
data_cfg.train_path = f"{DATA_PATH}/sentinel2/2021_seasons"

net_cfg: NetConfig = NET_CONFIG_S2_RESNET18
checkpoint_callback_cfg = CallbackConfig(
    class_name=ModelCheckpoint,
    args={
        'monitor': 'val_loss',
        'mode': 'min',
        'dirpath': f'{PACKAGE_PATH}/resources/models',
        'filename': f'{PROJECT}-{NAME}-' + '{epoch:02d}-{val_loss:.2f}',
        'save_top_k': 3
    }
)
stopping_callback_cfg = CallbackConfig(
    class_name=EarlyStopping,
    args={
        'monitor': 'val_loss',
        'patience': 3,
        'mode': 'min',
    }
)

net_cfg.callbacks = [checkpoint_callback_cfg, stopping_callback_cfg]


cfg: Config = Config(
    batch_size=6,
    num_workers=1,
    num_epochs=20,
    data_config=data_cfg,
    net_config=net_cfg,
    log_config=LogConfig(
        project=PROJECT,
        name=NAME,
        wandb_logger=True,
    )
)

train(cfg)  
# experiment = ExperimentBadResults("bad_results", cfg)
# experiment.run([('test', None)])

