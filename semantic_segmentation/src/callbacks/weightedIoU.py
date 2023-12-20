import os
from typing import Any
import numpy as np
from scipy.ndimage import distance_transform_edt
from pytorch_lightning.utilities.types import STEP_OUTPUT


from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import torch

from models.image_segmentation_module import ImageSegmentationModule
from data.data_processor import DataProcessor

class MeanWeightedIoU(Callback):
    
    def __init__(self, num_classes, alpha=1):        
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:

        if not isinstance(pl_module, ImageSegmentationModule):
            raise RuntimeError("WeightedIoU callback can only be used with ImageSegmentationModule")
        
        pl_module.rich_validation = True
        
    def on_validation_batch_end(self, trainer: Trainer, 
                                pl_module: LightningModule, 
                                outputs: STEP_OUTPUT | None, 
                                batch: Any, batch_idx: int, 
                                dataloader_idx: int = 0) -> None:
        
        masks = outputs["targets"].detach().cpu().numpy() 
        preds = outputs["preds"].detach().cpu().numpy()
        
        weights = list(map(self.compute_weight_mask, masks))

        IoUs = [self.wIoU(p, m, w) for p,m,w in zip(preds, masks, weights)]
        mean_wIoU = np.mean(IoUs)
        
        pl_module.log("val_mean_wIoU", mean_wIoU)

    def compute_weight_mask(self, mask):
    
        result = np.zeros(mask.shape)
        
        for c in range(self.num_classes):
            input_mask = np.zeros(mask.shape) 
            input_mask[mask == c] = 1
            
            dist = distance_transform_edt(input_mask)
            dist = dist / np.max(dist)
            
            result += dist
        
        return np.exp(-self.alpha * result)   
    
    def wIoU(self, pred, mask, weight):
        
        IoU = 0
        for c in range(self.num_classes):
        
            intersection = weight[np.logical_and(pred == c, mask ==c)].sum()
            union = weight[np.logical_or(pred == c, mask ==c)].sum()
            
            IoU += np.abs(intersection / union)
        
        wIoU = IoU / self.num_classes 
        
        return wIoU      
