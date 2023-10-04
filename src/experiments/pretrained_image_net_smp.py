import segmentation_models_pytorch as smp

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
import sys

sys.path.append('/home/daniel/Documents/hsi-forest-segmentation/Kyselica/src')

from config import *
from models.image_segmentation_module import ImageSegmentationModule
from utils import log_in_to_wandb

cfg: Config = Config(
    device='cuda',
    wandb_name=None,
    batch_size=32,
    num_workers=4,
    num_epochs=10,
    data_config=DataConfig(
        width=512,
        height=512,
        bands=[Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2], # RGB
        classes=set([ESAWorldCover.TREES]),
        stabilization_scale_factor=10_000,
        output_path='/home/daniel/Documents/hsi-forest-segmentation/Kyselica/resources/datasets',
        load=True, 
        train_path="/media/daniel/data/sentinel2/2021_seasons/Belgium_summer2021",
        val_path=None,
        grid_path=None,
        compute_mean_std=False,
        use_level_C1=False
    ),
    net_config=NetConfig(
        architecture=NetworkArchitectures.UNET,
        config={
            'encoder_name': 'timm-mobilenetv3_small_minimal_100',
            'encoder_weights': 'imagenet',
        } 
    )
)

log_in_to_wandb()

OPTIONS = list(NetworkArchitectures)

for i, option in enumerate(OPTIONS):
    print(f"\n=============== Running {option} ({i+1}/{len(OPTIONS)}) ===============\n")
    
    cfg.net_architecture = option
    
    wandb_logger = WandbLogger(
        project="ImageNet Pretrained",
        name=option,
        log_model="all"
    )

    model = ImageSegmentationModule(cfg)
    train_loader, val_loader = model.get_data_loaders()

    print(train_loader, val_loader)

    trainer = Trainer(max_epochs=cfg.num_epochs , logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)     
    
    wandb.finish()   