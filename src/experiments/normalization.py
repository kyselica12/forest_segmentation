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
    wandb_name=None,
    batch_size=16,
    num_workers=1,
    num_epochs=20,
    data_config=DataConfig(
        width=512,
        height=512,
        bands= ALL_BANDS_LIST,#[Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2], # RGB
        classes= set([ESAWorldCover.TREES]),
        stabilization_scale_factor=10_000,
        output_path='/home/daniel/Documents/hsi-forest-segmentation/Kyselica/resources/datasets',
        load=True, 
        train_path="/media/daniel/data/sentinel2/2021_seasons/Belgium_summer2021",
        val_path=None,
        grid_path=None,
        compute_mean_std=False,
        use_level_C1=True
    ),
    net_config=NetConfig(
        architecture=NetworkArchitectures.UNET,
        config={
            'encoder_name': CustomNets.RESNET18,
            'encoder_weights': CustomNetWeights.RESNET18,
        } 
    )
)

log_in_to_wandb()
register_SSL4EO_S12_encoders()

OPTIONS = [(True, "normalize"), (False, "no_normalize")] 

for i, option in enumerate(OPTIONS):
    print(f"\n=============== Running {option} ({i+1}/{len(OPTIONS)}) ===============\n")
    
    cfg.data_config.compute_mean_std = option[0]
    
    wandb_logger = WandbLogger(
        project="Normalization",
        name="Pretrained Resnet18 " + option[1],
        log_model="all"
    )

    model = ImageSegmentationModule(cfg)
    train_loader, val_loader = model.get_data_loaders()

    print(train_loader, val_loader)

    trainer = Trainer(max_epochs=cfg.num_epochs, logger=wandb_logger)
    trainer.fit(model, train_loader, val_loader)     
    
    wandb.finish()   