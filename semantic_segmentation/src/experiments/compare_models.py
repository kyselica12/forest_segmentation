import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import random
import sys
import os 
import re

sys.path.append(re.sub(r'/src/.*', '/src', os.path.abspath(__file__)))

import wandb

from configs.config import *
from configs.constants import *

from models.image_segmentation_module import ImageSegmentationModule
from data.data_processor import DataProcessor
from utils import train, get_wabdb_logger, register_SSL4EO_S12_encoders


RESULTS_PATH = f"{PACKAGE_PATH}/results/LR_vs_HR"
os.makedirs(RESULTS_PATH, exist_ok=True)

data_cfg_HR = DataConfig(
    train_path=f"{DATA_PATH}/sentinel2/dataset/HR_to_HR/docker",
    grid_path=f"{DATA_PATH}/sentinel2/dataset/HR_to_HR/docker/grid_features.json",
    output_path=f"{PACKAGE_PATH}/resources/datasets",
    classes= set([ESAWorldCover.TREES]),
    bands = [0,1,2],
    width=512,
    height=512,
    val_size=0.1,
    non_zero_ratio=0.5,
    first_id=1,
    stabilization_scale_factor=255,
    use_level_C1=False,
    improved_mask=False,
    compute_mean_std=False,
    load=True,
)

data_cfg_LR = DataConfig(
    train_path = f"{DATA_PATH}/sentinel2/dataset/multi_sensor_to_HR/docker",
    grid_path = f"{DATA_PATH}/sentinel2/dataset/multi_sensor_to_HR/docker/grid_features.json",
    output_path=f"{PACKAGE_PATH}/resources/datasets",
    classes= set([ESAWorldCover.TREES]),
    width = 128,
    height = 128,
    bands=ALL_BANDS_LIST,
    val_size = 0.1,
    first_id = 1,
    compute_mean_std = True,
    non_zero_ratio = 0.5,
    improved_mask = False,
    load = True,
    use_level_C1=True,
    stabilization_scale_factor=10_000,
)

register_SSL4EO_S12_encoders(PACKAGE_PATH)  

torch.manual_seed(42)

dp_HR = DataProcessor(data_cfg_HR)
dp_LR = DataProcessor(data_cfg_LR)

def get_id(path: str) -> str:
    name = os.path.split(path)[-1]
    return tuple(map(int, re.findall(r'(\d+)', name)))

random.seed(421)

_, val_set_HR = dp_HR.get_pytorch_datasets()
_, val_set_LR = dp_LR.get_pytorch_datasets()

sidx_HR = sorted(range(len(val_set_HR)), key=lambda x: get_id(val_set_HR.image_list[x]))  
sidx_LR = sorted(range(len(val_set_LR)), key=lambda x: get_id(val_set_LR.image_list[x]))  

val_set_HR.image_list = val_set_HR.image_list[sidx_HR]
val_set_HR.mask_list = val_set_HR.mask_list[sidx_HR]
val_set_LR.image_list = val_set_LR.image_list[sidx_LR]
val_set_LR.mask_list = val_set_LR.mask_list[sidx_LR]

print(val_set_LR.image_list[0], val_set_LR.mask_list[0])
print(val_set_HR.image_list[0], val_set_HR.mask_list[0])

MODELS_PATH = f"{PACKAGE_PATH}/resources/models/Super-Resolution"
HR_CKPTS = {
    "Cross Entropy Loss": f"{MODELS_PATH}/HR_CROSS_ENTROPY_LOSS/epoch=9-val_loss=0.19.ckpt",
    "IoU Loss": f"{MODELS_PATH}/HR_IOU_LOSS/epoch=21-val_loss=0.01.ckpt",
    "Focal Loss": f"{MODELS_PATH}/HR_FOCAL_LOSS/epoch=12-val_loss=0.09.ckpt",
}
LR_CKPTS = {
    "Cross Entropy Loss": f"{MODELS_PATH}/LR_CROSS_ENTROPY_LOSS/epoch=16-val_loss=0.19.ckpt",
    "IoU Loss": f"{MODELS_PATH}/LR_IOU_LOSS/epoch=16-val_loss=0.01.ckpt",
    "Focal Loss": f"{MODELS_PATH}/LR_FOCAL_LOSS/epoch=18-val_loss=0.08.ckpt",
}

models = {
    "HR": { k:ImageSegmentationModule.load_from_checkpoint(v) for k, v in HR_CKPTS.items() },
    "LR": { k:ImageSegmentationModule.load_from_checkpoint(v) for k, v in LR_CKPTS.items() },
}
[model.eval() for ms in models.values() for model in ms.values()]
data = {
    "HR": val_set_HR,
    "LR": val_set_LR,
}


def infer(model, val_set, idx):
    print(val_set.image_list[idx], val_set.mask_list[idx])
    d, l = val_set[idx]
    r = model(d.unsqueeze(0).cuda().float())
    preds, acc, loss = model._get_preds_acc_losses((d.unsqueeze(0).cuda().float(), l.unsqueeze(0).cuda().long()), 0)
    preds = preds[0].detach().cpu().numpy()
    loss = loss.mean().detach().item()
    labels = l.detach().cpu().numpy()
    labels[labels != 1] = 0
    acc = acc.detach().item()
    d = d.detach().cpu().numpy().transpose(1,2,0)
    return {
        "preds": preds,
        "acc": acc,
        "loss": loss,
        "d": d,
        "labels": labels,
    }

def mask_difference_img(preds, target):
    diff = preds - target
    diff_img = np.zeros((diff.shape[0], diff.shape[1], 3))
    diff_img[(diff == 0) * (target == 1)] = [0,1,0]
    diff_img[(diff == 1)] = [1,0,0]
    diff_img[(diff == -1)] = [0,0,1]
    return diff_img

def isEmptyTarget(idx, val_set):
    _, l = val_set[idx]
    return (l == 1).sum() == 0
    
    

# 20 random values from 0 to N
indices = random.sample(range(len(val_set_HR)), k=20)

for idx in indices:

    if isEmptyTarget(idx, val_set_HR):
        continue

    results = {
       resolution:{loss: infer(m, data[resolution], idx) for loss, m in ms.items()}for resolution, ms in models.items() 
    }

    LR_input = results["LR"]["Cross Entropy Loss"]["d"][:,:,[3,2,1]]
    LR_input = (LR_input - LR_input.min()) / (LR_input.max() - LR_input.min())
    HR_input = results["HR"]["Cross Entropy Loss"]["d"]
    target = results["HR"]["Cross Entropy Loss"]["labels"]

    
    fig, axs = plt.subplots(6,4, figsize=(4*5,6*5))

    axs[1, 0].imshow(HR_input)
    axs[1, 0].set_title("Super-resolution Input")
    axs[4, 0].imshow(LR_input)
    axs[4, 0].set_title("Multi-sensor Input")
    axs[1, 3].imshow(target, cmap="gray")
    axs[1, 3].set_title("Target")
    axs[4, 3].imshow(target, cmap="gray")
    axs[4, 3].set_title("Target")

    [a.axis('off') for ax in axs for a in ax]


    for start_i, resolution in [(0, "HR"), (3, "LR")]:
        for i, loss in enumerate(sorted(HR_CKPTS.keys())):
            i += start_i
            result = results[resolution][loss]
            axs[i, 1].imshow(color.label2rgb(result["preds"], HR_input, alpha=0.3))
            axs[i, 2].imshow(mask_difference_img(result["preds"], target))
            axs[i, 1].set_title(f"{loss}")
            axs[i, 2].set_title(f"Acc: {result['acc']*100:.2f}")
                
    plt.tight_layout()
    plt.axis('off')
        
    
    tile_id = get_id(val_set_HR.image_list[idx])
    
    plt.savefig(f"{RESULTS_PATH}/{tile_id}.png")


 
