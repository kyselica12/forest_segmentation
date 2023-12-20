import sys
import os 
from itertools import product

sys.path.append("/app")
sys.path.append("/app/src")

from models.image_segmentation_module import ImageSegmentationModule
from configs.config import DataConfig, NetConfig
from configs.config import *
from configs.constants import *
from utils import register_SSL4EO_S12_encoders, train_with_iterative_masks
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

logger = True # Default Logger -> WANDB downt work with interupting training

MASK_PATH = f"{DATA_PATH}/sentinel2/Itterative_masks/"
os.makedirs(MASK_PATH, exist_ok=True)

# weights = [0, 0.1, 0.5, 0.8]
weights = [0.1, 0.5, 0.8]
all_classes = [False]

STEP = 10
MAX_EPOCHS = 1*STEP

for w, c in product(weights, all_classes):

    CKPT_PATH = f"{DATA_PATH}/sentinel2/models/Itterative_masks/{w}"
    os.makedirs(CKPT_PATH, exist_ok=True)

    dp = DataProcessor(data_cfg)
    # module = ImageSegmentationModule(**net_cfg.__dict__)
    module = ImageSegmentationModule.load_from_checkpoint(f'{DATA_PATH}/sentinel2/models/checkpoint_0.ckpt')

    train_with_iterative_masks(module, dp,
                            batch_size=32,
                            num_workers=0,
                            step=STEP,
                            max_epochs=MAX_EPOCHS,
                            mask_path=MASK_PATH,
                            ckpt_path=CKPT_PATH,
                            weight=0.5,
                            diff_threshold=0.1,
                            pretrained=True,
                            logger=logger,
                            all_classes=c,
                            update_val_set=False)

