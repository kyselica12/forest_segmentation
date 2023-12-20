import os
import copy
import tifffile
import tqdm
import torch
from typing import List
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
import segmentation_models_pytorch as smp
import torchvision

from configs.config import CustomNets, CustomNetWeights 
from configs.constants import MASK_DIFFERENCE_INDEX, WANDB_API_KEY, PACKAGE_PATH, IGNORE_INDDEX
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

def get_wabdb_logger(project, name=None):
    log_in_to_wandb()
    return WandbLogger(
        project=project,
        name=name,
        log_model=False
    )

def train(module: ImageSegmentationModule, 
          data_processor: DataProcessor,
          num_epochs=10, 
          batch_size=16, 
          num_workers=4,
          callbacks=[],
          logger=None):


    if logger is not None and isinstance(logger, WandbLogger):
        # logger.experiment.config["data"] = data_processor.cfg.__dict__
        logger.log_hyperparams({"data": data_processor.cfg.__dict__})

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

    if logger is not None and isinstance(logger, WandbLogger):
        wandb.finish()


def generate_new_masks(module, dataset, all_classes, mask_path, diff_threshold=1):
    """Create new mask combining prediction and original. The differences are 
       marked with MASK_DIFFERENCE_INDEX - class label when mask[i] == class label and pred[i] != class label

    Args:
        module (ImageSegmenationModule)
        dataset (Dataset): _description_
        all_classes (Bool): If set differences will be marked for all classes, otherwise only for the class with label > 0
        mask_path : Path to the folder where the new masks will be saved

    Returns:
        List [str]: List of paths to the new masks
    """
    mask_files = []
    for i in tqdm.tqdm(range(len(dataset)), desc="Generating new masks"):
        x, y = dataset[i]
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        logits = module(x.float())
        y_hat = torch.argmax(logits, dim=1)

        y_new = y.clone()
        start = 0 if all_classes else 1
        for label in range(start, module.n_classes):
            diff = (y_hat != label) * (y == label) # Lablel i in the original mask, but not in the new one
            if diff.sum()/(y==label).sum() <= diff_threshold:
                y_new[diff] = MASK_DIFFERENCE_INDEX - label

        y_new_numpy = y_new.detach().cpu().numpy()
        y_new_numpy = y_new_numpy[0]

        name = os.path.split(dataset.mask_list[i])[1]
        new_path = f"{mask_path}/{name}"

        tifffile.imsave(new_path, y_new_numpy)
        dataset.mask_list[i] = new_path
        mask_files.append(new_path)

    return mask_files



def train_with_iterative_masks(module : ImageSegmentationModule,
                            data_processor: DataProcessor,
                            batch_size,
                            num_workers,
                            step,
                            max_epochs,
                            mask_path,
                            ckpt_path,
                            weight=0.5,
                            diff_threshold=1,
                            pretrained=False,
                            all_classes = True,
                            logger=None,
                            update_val_set=False):
    
    if logger is not None and isinstance(logger, WandbLogger):
        # logger.experiment.config["data"] = data_processor.cfg.__dict__
        logger.log_hyperparams({"data": data_processor.cfg.__dict__})

    train_set, val_set = data_processor.get_pytorch_datasets() 
    

    original_train_set = copy.deepcopy(train_set)
    original_val_set = copy.deepcopy(val_set)

    for i in range(0, max_epochs, step):
        if i > 0 or pretrained:
            module.improved_mask = True
            module.mask_difference_value = weight
            module.eval()

            train_masks = generate_new_masks(module, original_train_set, all_classes, mask_path, diff_threshold)
            train_set.mask_list = train_masks
            train_set.mapping_labels = None

            if update_val_set:
                val_masks = generate_new_masks(module, original_val_set, all_classes, mask_path)
                val_set.mask_list = val_masks
                val_set.mapping_labels = None

        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
        val_loader = DataLoader(val_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True)
        trainer = Trainer(
            default_root_dir=f"{PACKAGE_PATH}/resources/pl/",
            max_epochs=step,
            logger=logger,
        )
        module.train()
        trainer.fit(module, train_loader, val_loader)
        trainer.save_checkpoint(f"{ckpt_path}/checkpoint_{i}.ckpt")


    