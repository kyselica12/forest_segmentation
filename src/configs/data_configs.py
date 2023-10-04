from copy import deepcopy
from config import *


DEFAULT = DataConfig(
    val_size=0.2,
    load=True,
    bands= [Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2], # RGB
    classes= set([ESAWorldCover.TREES]),
    train_path=f"{DATA_PATH}/sentinel2/2021_seasons/Belgium_summer2021",
    compute_mean_std=True,
)

DATA_CONFIG_RGB = deepcopy(DEFAULT)
DATA_CONFIG_RGB.bands = [Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2]

DATA_CONFIG_S2_A = deepcopy(DEFAULT)
DATA_CONFIG_S2_A.bands = ALL_BANDS_LIST

DATA_CONFIG_S2_C1 = deepcopy(DATA_CONFIG_S2_A)
DATA_CONFIG_S2_C1.use_level_C1 = True



