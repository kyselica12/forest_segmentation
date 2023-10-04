import sys
import os 
import re

sys.path.append(re.sub(r'/src/.*', '/src', os.path.abspath(__file__)))

from config import *
from utils import train
from callbacks.save_results import SaveResults

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

cfg: Config = Config(
    batch_size=32,
    num_workers=1,
    num_epochs=20,
    data_config=DataConfig(
        val_size=0.2,
        load=True,
        bands= [Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2], # RGB
        classes= set([ESAWorldCover.TREES]),
        train_path=f"{DATA_PATH}/sentinel2/2021_seasons/Belgium_summer2021",
        compute_mean_std=True,
    ),
    net_config=NetConfig(
        architecture=NetworkArchitectures.UNET,
        config={
            'encoder_name': 'timm-mobilenetv3_small_minimal_100',
            'encoder_weights': 'imagenet',
        },
    ),
    log_config=LogConfig(
        project="Log bad examples",
        name="for epoch",
        log_model="all",
        wandb_logger=True,
        log_images=True,
        log_images_freq=Frequency.BATCH,
        n_images=1
    )
)

cfg.net_config.callbacks = [
    {
        "class": SaveResults,
        "args": {
            "num": 3,
            "path": f"{PACKAGE_PATH}/resources/{cfg.log_config.project}/{cfg.log_config.name}", 
            "frequency": "epoch",
            "monitor": "loss",
            "mode": "max", 
            "size": 4  
        }
    }
]

print(cfg.data_config.train_path)
train(cfg)
