#from logging import log
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim

import segmentation_models_pytorch as smp

from configs.config import NetConfig, NetworkArchitectures
from configs.constants import IGNORE_INDDEX, MASK_DIFFERENCE_INDEX

class ImageSegmentationModule(pl.LightningModule):
    
    def __init__(self, n_classes, in_channels, architecture, args):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.architecture = architecture
        self.net_args = args
        self.rich_validation = False

        self.save_hyperparameters()

        self.net = self._initialize_net()
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDDEX)
        torch.set_float32_matmul_precision("high")
        
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        preds, acc, losses = self._get_preds_acc_losses(batch, batch_idx)
        loss = losses.mean()
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        self.log('train_dice_score', self._dice_score(preds, batch[1]), sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, acc, losses = self._get_preds_acc_losses(batch, batch_idx)
        loss = losses.mean()
        self.log('val_loss', loss,  sync_dist=True)
        self.log('val_acc', acc,  sync_dist=True)
        self.log('val_dice_score', self._dice_score(preds, batch[1]),  sync_dist=True)

        if self.rich_validation:
            return {
                "inputs": batch[0],
                "targets": batch[1],
                "preds": preds,
                "metric": losses
            } 
        
        return loss

    def test_step(self, batch, batch_idx):
        preds, acc, losses = self._get_preds_acc_losses(batch, batch_idx)
        loss = losses.mean()
        self.test_step_outputs.append({
            'preds': preds, 'acc': acc, 'loss': loss,
            'input': batch[0], 'target': batch[1]
        }) 
        
        return loss
       
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        
        return optimizer
    
    def _get_preds_acc_losses(self, batch, batch_idx):
        x, y = batch
        #TODO if y contains MASK_DIFFERENCE_INDEX create weight map for the loss function
        # target = y.copy()
        # weight_map = torch.ones_like(target)
        # weight_map[target == MASK_DIFFERENCE_INDEX] = 0.5 # value to be set in the config
        # target[target == MASK_DIFFERENCE_INDEX] = 1
        
        logits = self.net(x.float())
        losses = self.criterion(logits, y.long())

        # losses = losses * weight_map

        preds = torch.argmax(logits,dim=1)
        acc = (preds == y).sum() / torch.numel(preds)
     
        return preds, acc, losses

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
    
    
         