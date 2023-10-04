import os
import time
import heapq
from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning.utilities.types import STEP_OUTPUT

from skimage import color

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import torch

from models.image_segmentation_module import ImageSegmentationModule


class SaveResults(Callback):
    
    def __init__(self, num=1, path="./", frequency="epoch", monitor="loss", mode="max", size=2) -> None:
        super().__init__()

        self.num = num
        self.path = path
        self.frequency = frequency
        self.monitor = monitor
        self.mode = mode
        self.size = size
        self.queue = []

        self.default_value = -1e9
        
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:

        if not isinstance(pl_module, ImageSegmentationModule):
            raise RuntimeError("SaveResults callback can only be used with ImageSegmentationModule")
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        pl_module.rich_validation = True
    
    def on_validation_batch_end(self, trainer: Trainer, 
                                pl_module: LightningModule, 
                                outputs: STEP_OUTPUT | None, 
                                batch: Any, batch_idx: int, 
                                dataloader_idx: int = 0) -> None:
        
        indices = torch.argsort(outputs["metric"], descending=self.mode == "max")
        
        self.add(outputs["inputs"], outputs["targets"], outputs["preds"], outputs["metric"], indices)
        
        if self.frequency == "batch":
            self.save_queue(trainer, pl_module) 
            self.queue = []
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: ImageSegmentationModule):
        if self.frequency == "epoch":
            self.save_queue(trainer, pl_module) 
            self.queue = []

    def add(self, inputs, targets, preds, metric, indices):
        
        current = self.queue[0][0] if len(self.queue) > 0 else self.default_value
         
        for i in indices[:self.num]:
            if metric[i] > current:
                heapq.heappush(
                    self.queue,
                    (
                        metric[i] * (1 if self.mode == "max" else -1), 
                        { 
                        self.monitor: metric[i], 
                        "preds": preds[i].detach().cpu().numpy(),
                        "input": inputs[i].detach().cpu().numpy(), 
                        "target": targets[i].detach().cpu().numpy()
                        }
                    )
                )
            
                if len(self.queue) > self.num:
                    heapq.heappop(self.queue)
             
                current = self.queue[0][0]
    
    def save_queue(self, trainer: Trainer, module: ImageSegmentationModule):
        
        columns = ["metric", "input", "pred", "target", "diff"]
        data = []
        
        for metric, images in self.queue:
            
            img = module.data_processor.create_rgb_image(images["input"])
            # print(img)
            # print(images["preds"].shape, images["target"].shape)
            data.append(
                [
                    metric,
                    img,
                    color.label2rgb(images["preds"], img, alpha=0.3),
                    color.label2rgb(images["target"], img,  alpha=0.3),
                    module.data_processor.create_mask_difference_image(images["target"], images["preds"], img) 
                ]
            )    
                
        epoch = trainer.current_epoch
        
        width = len(columns) - 1
        fig, axs = plt.subplots(len(data), width)
        fig.set_size_inches(width*self.size, len(data)*self.size)
        
        for i, example in enumerate(data):
            metric = example[0]
            for j, img in enumerate(example[1:]):
                axs[i, j].imshow(img)
                axs[i, j].set_title(columns[j+1])
                axs[i, j].get_yaxis().set_ticks([])
                axs[i, j].get_xaxis().set_ticks([])
                axs[i, j].set_ylabel(f"{metric:.4f}")
        
        plt.savefig(f"{self.path}/epoch_{epoch}_{self.monitor}_{int(time.time()*1000)}.png")
        plt.close(fig)    