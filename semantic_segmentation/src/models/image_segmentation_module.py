#from logging import log
from numpy import mat
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import focal_loss

import segmentation_models_pytorch as smp

from configs.config import LossConfig, NetConfig, NetworkArchitectures
from configs.constants import CROSS_ENTROPY_LOSS, IGNORE_INDDEX, MASK_DIFFERENCE_INDEX
from models.loss_functions import CrossEntropyLoss, FocalLoss, IoULoss

class ImageSegmentationModule(pl.LightningModule):
    
    def __init__(self, n_classes, in_channels, architecture, loss_cfg=None, upscale_ratio=1, args=None):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.architecture = architecture
        self.net_args = args
        self.rich_validation = False
        self.upscale_ratio = upscale_ratio

        self.save_hyperparameters()

        self.net = self._initialize_net()
        loss_cfg = loss_cfg if loss_cfg is not None else CROSS_ENTROPY_LOSS
        self.criterion = self._initialize_loss(loss_cfg)
        torch.set_float32_matmul_precision("high")

        self.improved_mask = False
        self.mask_difference_value = 0.5
        
        
    def forward(self, x):
        logits = self.net(x)
        if self.upscale_ratio != 1:
            logits = torch.nn.functional.interpolate(logits, scale_factor=self.upscale_ratio, mode='bilinear', align_corners=True)
        return logits

    
    def training_step(self, batch, batch_idx):
        # print((batch[1] == MASK_DIFFERENCE_INDEX).sum())
        if self.improved_mask:
            weight_map = self._get_improved_mask_weight_map(batch[1], self.mask_difference_value)
        else:
            weight_map = None
        
        preds, acc, loss = self._get_preds_acc_losses(batch, batch_idx, weight_map)
        # print(losses.mean() - (losses * weight_map).mean(), losses.shape, weight_map.shape, batch[1][batch[1] == MASK_DIFFERENCE_INDEX].sum())

        self.log('train_loss', loss, on_step=True)
        self.log('train_acc', acc, on_step=True)
        self.log('train_dice_score', self._dice_score(preds, batch[1]), on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.improved_mask:
            weight_map = self._get_improved_mask_weight_map(batch[1], self.mask_difference_value)
        else:
            weight_map = None
            
        preds, acc, loss = self._get_preds_acc_losses(batch, batch_idx, weight_map)
        
        self.log('val_loss', loss,  on_step=True)
        self.log('val_acc', acc,  on_step=True)
        self.log('val_dice', self._dice_score(preds, batch[1]),  on_step=True)

        if self.rich_validation:
            return {
                "inputs": batch[0],
                "targets": batch[1],
                "preds": preds,
                "metric": None #FIXME broke some callbacks
            } 
        
        return loss

    def test_step(self, batch, batch_idx):
        if self.improved_mask:
            weight_map = self._get_improved_mask_weight_map(batch[1], self.mask_difference_value)
        else:
            weight_map = None
        preds, acc, loss = self._get_preds_acc_losses(batch, batch_idx, weight_map)

        return loss
       
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
        
        # return optimizer
    
    def _get_preds_acc_losses(self, batch, batch_idx, weight=None):
        x, y = batch
        
        logits = self.net(x.float())

        # if the target is in higher resolution than the input, interpolate the logits
        if y.shape[1] != x.shape[2] and y.shape[2] != x.shape[3]:
            logits = torch.nn.functional.interpolate(logits, size=(y.shape[1], y.shape[2]), mode='bilinear', align_corners=True)

        losses = self.criterion.forward(logits, y.long(), weight)

        preds = torch.argmax(logits,dim=1)
        acc = (preds == y).sum() / torch.numel(preds)
        
        # if batch_idx %100 == 0:
            # print(torch.sum(preds == 1), torch.sum(y == 1), acc)
     
        return preds, acc, losses
    
    def _get_improved_mask_weight_map(self, y, value):
        
        weight_map = torch.ones_like(y).type(torch.float32)

        for i in range(self.n_classes):
            IDX = MASK_DIFFERENCE_INDEX - i
            ok = y == IDX
            weight_map[ok] = value # value to be set in the config
            y[ok] = i

        return weight_map

    def _dice_score(self, preds, targets):
        
        dice_score = 0
        for i in range(0, self.n_classes):
            intersection = torch.logical_and(preds == i, targets == i).sum()
            union = (preds == i).sum() + (targets == i).sum()
            dice = 2 * (intersection +1) / (union + 1)
            dice_score += dice
        
        dice_score /= self.n_classes
                 
        return dice_score
    
    def get_data_loaders(self):
        """Initialize data loaders using main configuration

        Returns:
            train_loader: DataLoader, val_loader: DataLoader
        """
        
        train_loader, val_loader = self.data_processor.get_pytorch_datasets() 
        
        return train_loader, val_loader
    
    def _initialize_loss(self, loss_cfg: LossConfig):
        match loss_cfg.name:
            case "CrossEntropyLoss":
                return CrossEntropyLoss(**loss_cfg.args)
            case "IoULoss":
                return IoULoss(**loss_cfg.args)
            case "FocalLoss":
                return FocalLoss(**loss_cfg.args)
            case _:
                raise("Unknown loss function")
            
    
    def _initialize_net(self):
        """Innitialize network using main configuration
        
        Pretrained model fom segmentation_models_pytorch
        All available models in config.py -> NetworkArchitectures

        Returns:
            nn.Module: initialized model
        """
        
        match self.architecture:
            case NetworkArchitectures.DEEPLABV3:
                net_class = smp.DeepLabV3
            case NetworkArchitectures.DEEPLABV3PLUS:
                net_class = smp.DeepLabV3Plus
            case NetworkArchitectures.UNET:
                net_class = smp.Unet
            case NetworkArchitectures.UNETPLUSPLUS:
                net_class = smp.UnetPlusPlus
            case NetworkArchitectures.PAN:
                net_class = smp.PAN
            case NetworkArchitectures.MANET:
                net_class = smp.MAnet
            case NetworkArchitectures.LINKNET:
                net_class = smp.Linknet
            case NetworkArchitectures.FPN:  
                net_class = smp.FPN
            case NetworkArchitectures.PSPNET:
                net_class = smp.PSPNet
            case _:
                net_class = None
        
        if net_class is None:
            raise("Unknown network architecture")
        
        net = net_class(
                        in_channels=self.in_channels, 
                        classes=self.n_classes,
                        **self.net_args
                        )
        
        return net
    
    
         