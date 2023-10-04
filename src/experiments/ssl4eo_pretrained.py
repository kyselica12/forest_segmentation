import segmentation_models_pytorch as smp

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
import sys

sys.path.append('/home/daniel/Documents/hsi-forest-segmentation/Kyselica/src')

from config import *
from models.image_segmentation_module import ImageSegmentationModule
from utils import log_in_to_wandb, register_SSL4EO_S12_encoders

cfg: Config = Config(
    device='cuda',
    width=512,
    height=512,
    bands=[Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2],
    classes=set([ESAWorldCover.TREES]),
    stabilization_scale_factor=10_000,
    wandb_name=None,
    batch_size=16,
    num_workers=1,
    num_epochs=10,
    dataset_path='/home/daniel/Documents/hsi-forest-segmentation/Kyselica/resources',
    image_path=None,
    mask_path=None,
    net_architecture=NetworkArchitectures.UNET,
    net_config={
        'encoder_name': '',
        'encoder_weights': 'imagenet',
    },
    use_level_C1=False,
)

register_SSL4EO_S12_encoders()
log_in_to_wandb()

OPTIONS = ['resnet18', 'resnet50']

for i, option in enumerate(OPTIONS):
    print(f"\n=============== Running {option} ({i+1}/{len(OPTIONS)}) ===============\n")
    
    cfg.net_config['encoder_name'] = option
    
    wandb_logger = WandbLogger(
        project="SSL4EO-S12 Pretrained",
        name=option + '_imagenet',
        log_model="all"
    )

    model = ImageSegmentationModule(cfg)
    train_loader, val_loader = model.get_data_loaders()

    trainer = Trainer(max_epochs=cfg.num_epochs, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)     
    
    wandb.finish()
