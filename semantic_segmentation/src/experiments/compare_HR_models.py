import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import random
import os 
import re
from typing import Tuple
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

sys.path.append(re.sub(r'/src/.*', '/src', os.path.abspath(__file__)))

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from configs.config import *
from configs.constants import *

from models.image_segmentation_module import ImageSegmentationModule
from data.data_processor import DataProcessor

OUTPUT_PATH = f"{PACKAGE_PATH}/results/Super-Resolution/loss_functions"
os.makedirs(OUTPUT_PATH, exist_ok=True)

data_cfg = DataConfig(
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

net_cfg: NetConfig = NET_CONFIG_IMAGENET_RESNET18

MODEL_PATH = f'{PACKAGE_PATH}/resources/models/Super-Resolution/'
loss_function = [
    ("HR_CROSS_ENTROPY_LOSS", 9, 0.19),
    ("HR_FOCAL_LOSS", 4, 0.09),
    ("HR_IOU_LOSS", 21, 0.01),
]

models = [
    ImageSegmentationModule.load_from_checkpoint(f"{MODEL_PATH}/{desc}/epoch={epoch}-val_loss={loss:.2f}.ckpt")
    for desc, epoch, loss in loss_function
]
[module.eval() for module in models]

torch.manual_seed(42)

dp = DataProcessor(data_cfg)


_, val_set = dp.get_pytorch_datasets()

def get_id(path: str) -> str:
    name = os.path.split(path)[-1]
    return tuple(map(int, re.findall(r'(\d+)', name)))


sidx = sorted(range(len(val_set)), key=lambda x: get_id(val_set.image_list[x]))  

val_set.image_list = val_set.image_list[sidx]
val_set.mask_list = val_set.mask_list[sidx]


# 20 random values from 0 to N
random.seed(421)
indices = random.sample(range(len(val_set)), k=20)


for idx in indices:
    def infer(model, val_set, idx):
        def toNumpy(x):
            return x.detach().cpu().numpy()

        print(val_set.image_list[idx], val_set.mask_list[idx])
        d, l = val_set[idx]
        r = model(d.unsqueeze(0).cuda().float())
        preds, acc, loss = model._get_preds_acc_losses((d.unsqueeze(0).cuda().float(), l.unsqueeze(0).cuda().long()), 0)
        loss = loss.mean().detach().item()
        labels = toNumpy(l)
        labels[labels != 1] = 0
        d = toNumpy(d)
        acc = acc.detach().item()
        preds = toNumpy(preds).squeeze()


        return preds, acc, loss, d, labels
    
    results = [infer(m, val_set, idx) for m in models]

        
    print(results[0][-2].shape, results[0][0].shape)    
    orig_img = results[0][-2].transpose(1,2,0)
    labels = results[0][-1]
    label_image = np.zeros((labels.shape[0], labels.shape[1], 3))
    label_image[labels == 1] = [1,1,1]

    if np.sum(labels == 1) == 0:
        continue

    img_target = color.label2rgb(labels, orig_img, alpha=0.3)

    images = [orig_img] + [color.label2rgb(preds,orig_img, alpha=0.3) for preds, _, _, _, _ in results]

    def diffImage(preds, labels):
        diff = preds - labels
        diff_img = np.zeros((diff.shape[0], diff.shape[1], 3))
        diff_img[(diff == 0) * (labels == 1)] = [0,1,0]
        diff_img[(diff == 1)] = [1,0,0]
        diff_img[(diff == -1)] = [0,0,1]
        return diff_img
    diff = [label_image] + [diffImage(preds, labels) for preds, _,_, _, labels in results]

    names = ["Target"] + [desc for desc, _, _ in loss_function]
    accuracies = ["Accuracy"] + [acc for _,acc,_,_,_ in results]

    fig, axs = plt.subplots(len(images), 2, figsize=(10, 5*len(images)))

    for i, (img, dif_img) in enumerate(zip(images, diff)):
        axs[i,0].imshow(img)
        axs[i,0].set_title(names[i])
        axs[i,1].imshow(dif_img)
        axs[i,1].set_title(f"{accuracies[i]}") 

    plt.tight_layout()
    plt.axis('off')
    
    tile_id = get_id(val_set.image_list[idx])
    plt.savefig(f"{OUTPUT_PATH}/{tile_id}.png")


 
