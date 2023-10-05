from typing import List
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
import segmentation_models_pytorch as smp
import torchvision

from configs.config import CustomNets, CustomNetWeights 
from configs.constants import WANDB_API_KEY, PACKAGE_PATH
from models.backbones.resnet_encoder import SentinelResNetEncoder
from models.image_segmentation_module import ImageSegmentationModule
from data.data_processor import DataProcessor


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

def get_wabdb_logger(project, name=None, log_model='all'):
    log_in_to_wandb()
    return WandbLogger(
        project=project,
        name=name,
        log_model=log_model
    )

def train(module: ImageSegmentationModule, 
          data_processor: DataProcessor,
          num_epochs=10, 
          batch_size=16, 
          num_workers=4,
          callbacks=[],
          logger=None):

    register_SSL4EO_S12_encoders(PACKAGE_PATH)
    
    train_set, val_set = data_processor.get_pytorch_datasets() 

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
    )
    
    trainer = Trainer(default_root_dir=f"{PACKAGE_PATH}/resources/pl/", max_epochs=num_epochs, logger=logger, callbacks=callbacks)

    trainer.fit(module, train_loader, val_loader)     

    