from dataclasses import dataclass, field
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from enum import IntEnum
from strenum import StrEnum
import os

class Sentinel2Bands(IntEnum):
    """Sentinel 2 Bands"""""
    B1 = 0      # Aerosols
    B2 = 1      # Blue 
    B3 = 2      # Green
    B4 = 3      # Red
    B5 = 4      # Red Edge 1
    B6 = 5      # Red Edge 2
    B7 = 6      # Red Edge 3
    B8 = 7      # NIR
    B8A = 8     # Red Edge 4
    B9 = 9      # Water Vapor
    B11 = 10    # SWIR 1
    B12 = 11    # SWIR 2
    

class ESAWorldCover(IntEnum):
    """ESA World Cover Land Cover Classes"""""
    TREES = 10
    SHRUBLAND = 20
    GRASSLAND = 30
    CROPLAND = 40
    BUILTUP = 50
    BARE = 60
    SNOW = 70
    WATER = 80
    WETLAND = 90
    MANGROVES = 95
    MOSS = 100


ESAWorldCoverColors = {
    ESAWorldCover.TREES: "#006400",
    ESAWorldCover.SHRUBLAND: "#ffbb22",
    ESAWorldCover.GRASSLAND: "#ffff4c",
    ESAWorldCover.CROPLAND: "#f096ff",
    ESAWorldCover.BUILTUP: "#fa0000",
    ESAWorldCover.BARE: "#b4b4b4",
    ESAWorldCover.SNOW: "#f0f0f0",
    ESAWorldCover.WATER: "#0064c8",
    ESAWorldCover.WETLAND: "#0096a0",
    ESAWorldCover.MANGROVES: "#00cf75",
    ESAWorldCover.MOSS: "#fae6a0"
}

@dataclass
class DataConfig:
    width: int = 512                           # width of the image
    height: int = 512                          # height of the image
    bands: list = field(default_factory=list)  # list of bands to use -> Just indices of the bands
    classes: set = field(default_factory=set)  # set of classes to use
    stabilization_scale_factor: int = 10000    # divide image by this factor -> 10 000 for Sentinel 2
    batch_size: int = 32                       
    num_workers: int = 4
    val_size: float = 0.2
    random_state: int = 42
    output_path: str = ""
    train_path: str = None                     # path to the training data
    val_path: str = None                       # path to the validation data -> if None train data are split
    grid_path: str = None                      # path to the grid features
    load: bool = True                          # load the data
    compute_mean_std: bool = False             
    use_level_C1: bool = False                 # adds one empty channel to the A2 Sentinel 2 data
    non_zero_ratio: float = 0.5                # minimum ratio of non zero pixels in the image
    improved_mask: bool = False                # mask contains special indices
    first_id: int = 0                          # first id of the images

class LossFunctions(StrEnum):
    IOU = 'IoULoss'
    FOCAL = 'FocalLoss'
    CROSS_ENTROPY = 'CrossEntropyLoss'

@dataclass
class LossConfig:
    name: str = LossFunctions.CROSS_ENTROPY
    args: dict = field(default_factory=dict)
    

@dataclass
class NetConfig:
    architecture: str = 'DeepLabV3Plus'
    args: dict = field(default_factory=dict)
    in_channels: int = 3
    n_classes: int = 2
    loss_cfg: LossConfig = field(default_factory=LossConfig)
    upscale_ratio: int = 1

   
class NetworkArchitectures(StrEnum):
    DEEPLABV3 = 'DeepLabV3'
    DEEPLABV3PLUS = 'DeepLabV3Plus'
    UNET = 'Unet'
    UNETPLUSPLUS = 'UnetPlusPlus'
    PAN = 'PAN'
    MANET = 'MAnet'
    LINKNET = 'Linknet'
    FPN = 'FPN'
    PSPNET = 'PSPNet'
    
class CustomNets(StrEnum):
    RESNET18 = 'resnet18_S2'
    RESNET50 = 'resnet50_S2'
    
class CustomNetWeights(StrEnum):
    RESNET18 = "SSL4EO-S12"
    RESNET50 = "SSL4EO-S12"
