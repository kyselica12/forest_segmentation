#from logging import log
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim

import segmentation_models_pytorch as smp

from configs.config import NetConfig, NetworkArchitectures

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
        self.criterion = nn.CrossEntropyLoss()
        torch.set_float32_matmul_precision("high")
        
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        preds, acc, loss = self._get_preds_acc_loss(batch, batch_idx)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', acc, sync_dist=True)
        self.log('train_dice_score', self._dice_score(preds, batch[1]), sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.rich_validation:
            return self._rich_validation_step(batch, batch_idx)

        return self._validation_step_basic(batch, batch_idx)
        
    def _validation_step_basic(self, batch, batch_idx):
        preds, acc, loss = self._get_preds_acc_loss(batch, batch_idx)
        self.log('val_loss', loss,  sync_dist=True)
        self.log('val_acc', acc,  sync_dist=True)
        self.log('val_dice_score', self._dice_score(preds, batch[1]),  sync_dist=True)
        return loss
    
    def _rich_validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x.float())
        preds = torch.argmax(logits,dim=1)
        acc = (preds == y).sum() / torch.numel(preds)
        
        losses = []
        for l, t in zip(logits, y):
            l = l.unsqueeze(0)
            t = t.unsqueeze(0)
            losses.append(self.criterion(l, t.long()))
        
        losses = torch.tensor(losses)
        
        self.log('val_loss', losses.mean(), sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_dice_score', self._dice_score(preds, batch[1]), sync_dist=True)

        return {
            "inputs": batch[0],
            "targets": batch[1],
            "preds": preds,
            "metric": losses
        } 

    def test_step(self, batch, batch_idx):
        preds, acc, loss = self._get_preds_acc_loss(batch, batch_idx)
        
        self.test_step_outputs.append({
            'preds': preds, 'acc': acc, 'loss': loss,
            'input': batch[0], 'target': batch[1]
        }) 
        
        return loss
       
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def _get_preds_acc_loss(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x.float())
        loss = self.criterion(logits, y.long())
        preds = torch.argmax(logits,dim=1)
        acc = (preds == y).sum() / torch.numel(preds)
     
        return preds, acc, loss

    def _dice_score(self, preds, targets):
        
        dice_score = 0
        for i in range(self.n_classes):
            intersection = torch.logical_and(preds == (i+1), targets == (i+1)).sum()
            union = (preds == (i+1)).sum() + (targets == (i+1)).sum()
            dice = 2 * intersection / union
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
    
    
         