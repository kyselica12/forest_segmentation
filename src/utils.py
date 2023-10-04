from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import wandb
import segmentation_models_pytorch as smp
import torchvision

from config import CustomNets, CustomNetWeights, Config, PACKAGE_PATH
from models.backbones.resnet_encoder import SentinelResNetEncoder
from models.image_segmentation_module import ImageSegmentationModule

WANDB_API_KEY = "b73e051ec86e9d3e56a2d2c47f1e3661a1b2a4db"

def log_in_to_wandb():
    try:
        api_key = WANDB_API_KEY
        wandb.login(key=api_key)
        anonymous = None
    except:
        anonymous = "must"
        print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')

def register_SSL4EO_S12_encoders(path):
    
    smp.encoders.encoders[CustomNets.RESNET50] = {
        'encoder': SentinelResNetEncoder,
        'pretrained_settings': {
            CustomNetWeights.RESNET50: {
                'url': f'file://{path}/resources/backbones/resnet50.pth',
            }
        },
        'params': {
            'out_channels': (13, 64, 256, 512, 1024, 2048),
            'block': torchvision.models.resnet.Bottleneck,
            'layers': [3, 4, 6, 3],
            'depth': 5,
        }        
    }
    
    smp.encoders.encoders[CustomNets.RESNET18] = {
        'encoder': SentinelResNetEncoder,
        'pretrained_settings': {
            CustomNetWeights.RESNET18: {
                'url': f'file://{path}/resources/backbones/resnet18.pth',
            }
        },
        'params': {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": torchvision.models.resnet.BasicBlock,
            "layers": [2, 2, 2, 2],
        }    
    }

def get_callbacks(callback_cfg):
    
    callbacks = []
    for callback in callback_cfg:
        callbacks.append(callback["class"](**callback["args"]))
    
    return callbacks
        
def train(config: Config):
    logger = False

    register_SSL4EO_S12_encoders(PACKAGE_PATH)
    if config.log_config.wandb_logger:
        log_in_to_wandb()
        logger = WandbLogger(
                project=config.log_config.project,
                name=config.log_config.name,
                log_model=config.log_config.log_model
        )

    model = ImageSegmentationModule(config)
    train_loader, val_loader = model.get_data_loaders()

    trainer = Trainer(
        default_root_dir=f"{PACKAGE_PATH}/resources/pl/",
        max_epochs=config.num_epochs, 
        logger=logger, 
        callbacks=get_callbacks(config.net_config.callbacks)
    )

    trainer.fit(model, train_loader, val_loader)     

    if config.log_config.wandb_logger:
        wandb.finish()

    