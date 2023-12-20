import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import random
import sys

sys.path.append("/app")
sys.path.append("/app/src")

from models.image_segmentation_module import ImageSegmentationModule
from configs.config import DataConfig, NetConfig
from configs.config import *
from configs.constants import *
from data.dataset import SentinelDataset
from utils import register_SSL4EO_S12_encoders, train_with_iterative_masks, get_wabdb_logger
from data.data_processor import DataProcessor

data_cfg = DataConfig(
    train_path = f"{DATA_PATH}/sentinel2/dataset/multi_sensor_10m/docker",
    grid_path = f"{DATA_PATH}/sentinel2/dataset/multi_sensor_10m/docker/grid_features.json",
    output_path=f"{PACKAGE_PATH}/resources/datasets",
    classes= set([ESAWorldCover.TREES]),
    width = 512,
    height = 512,
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
net_cfg: NetConfig = NET_CONFIG_S2_RESNET18
net_cfg.loss_cfg = CROSS_ENTROPY_LOSS
net_cfg.upscale_ratio = 1

RESULT_PATH = f"{PACKAGE_PATH}/results/Itterative_masks/"
os.makedirs(RESULT_PATH, exist_ok=True)

# CKPT_PATH = f"{PACKAGE_PATH}/resources/models/Itterative_masks/"
CKPT_PATH = f"{DATA_PATH}/sentinel2/models/Itterative_masks/"


paths = [
    f"{DATA_PATH}/sentinel2/models/checkpoint_0.ckpt",
    f"{CKPT_PATH}/0/checkpoint_0.ckpt",
    f"{CKPT_PATH}/0.1/checkpoint_0.ckpt",
    f"{CKPT_PATH}/0.5/checkpoint_0.ckpt",
    f"{CKPT_PATH}/0.8/checkpoint_0.ckpt",
]




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

dp = DataProcessor(data_cfg)
val_set = dp.get_pytorch_datasets()[0]

random.seed(421)
N = 10

indices = list(range(len(val_set.mask_list)))
random.shuffle(indices)

register_SSL4EO_S12_encoders(PACKAGE_PATH)  
for path, w in zip(paths, ["Base",0, 0.1, 0.5, 0.8]):
    module = ImageSegmentationModule.load_from_checkpoint(path)
    module.cuda()
    module.eval()

    c = N

    fig, axs = plt.subplots(N, 3, figsize=(15, 5*N))
    [a.axis('off') for ax in axs for a in ax]

    for i, idx in enumerate(indices):
        if c == 0: 
            break
        c -= 1
        if isEmptyTarget(idx, val_set):
            continue 

        res = infer(module, val_set, idx) 

        input_img = res["d"][:,:,[3,2,1]]
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

        axs[i, 0].imshow(input_img)
        axs[i, 1].imshow(color.label2rgb(res["preds"], input_img, alpha=0.3))
        axs[i, 2].imshow(color.label2rgb(res["labels"], input_img, alpha=0.3))

    plt.tight_layout()
    plt.axis('off')

    plt.savefig(f"{RESULT_PATH}/{str(w).replace('.','-')}_results.png")
        