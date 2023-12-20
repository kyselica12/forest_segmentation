import sys
import os 
import re
from typing import Tuple
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


sys.path.append("/app")
sys.path.append("/app/src")

from configs.config import DataConfig, NetConfig
from configs.config import *
from configs.constants import *
from utils import register_SSL4EO_S12_encoders

from experiment import Experiment


class LossFunctionExperiment(Experiment):

    def process_option(self, desc, val) -> Tuple[DataConfig, NetConfig]:
        torch.manual_seed(42)
        data_cfg, net_cfg =  super().process_option(desc, val)
        net_cfg.loss_cfg = val
        return data_cfg, net_cfg
    
    def build_module(self, desc, val, net_cfg):
        module =  super().build_module(desc, val, net_cfg)
        module.improved_mask = False
        module.mask_difference_value = 1
        return module
    
    def build_callbacks(self, desc, val):
        ckpt_dir = f'{PACKAGE_PATH}/resources/models/{self.name}/{desc}'
        os.makedirs(ckpt_dir, exist_ok=True)
        return [
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename='{epoch}-{val_loss:.2f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=2,
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min',
                )
        ]

PROJECT = "Super-Resolution"

data_cfg = DataConfig(
    train_path = f"{DATA_PATH}/sentinel2/dataset/multi_sensor_to_HR/docker",
    grid_path = f"{DATA_PATH}/sentinel2/dataset/multi_sensor_to_HR/docker/grid_features.json",
    output_path=f"{PACKAGE_PATH}/resources/datasets",
    classes= set([ESAWorldCover.TREES]),
    width = 128,
    height = 128,
    bands=ALL_BANDS_LIST,
    val_size = 0.1,
    first_id = 1,
    compute_mean_std = True,
    non_zero_ratio = 0.5,
    improved_mask = False,
    load = True,
    use_level_C1=True,
    stabilization_scale_factor=10_000,
)

register_SSL4EO_S12_encoders(PACKAGE_PATH)  
net_cfg: NetConfig = NET_CONFIG_S2_RESNET18
net_cfg.loss_cfg = IOU_LOSS

OPTIONS = [
    ("LR_CROSS_ENTROPY_LOSS", CROSS_ENTROPY_LOSS),
    ("LR_FOCAL_LOSS", FOCAL_LOSS),
    ("LR_IOU_LOSS", IOU_LOSS),
]

e = LossFunctionExperiment(PROJECT, data_cfg, net_cfg, log_to_wandb=True)
e.run(OPTIONS, 1000, 32, 0)
