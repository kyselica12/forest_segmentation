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
                    save_top_k=3,
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min',
                )
        ]

    

PROJECT = "Super-Resolution"

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

OPTIONS = [
    ("HR_CROSS_ENTROPY_LOSS", CROSS_ENTROPY_LOSS),
    ("HR_FOCAL_LOSS", FOCAL_LOSS),
    ("HR_IOU_LOSS", IOU_LOSS),
]

e = LossFunctionExperiment(PROJECT, data_cfg, net_cfg, log_to_wandb=True)
e.run(OPTIONS, 1000, 32, 0)





