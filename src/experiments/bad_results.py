import sys
import os 
import re

sys.path.append(re.sub(r'/src/.*', '/src', os.path.abspath(__file__)))

from config import *
from configs.network_configs import *
from configs.data_configs import *

from experiments.experiment import Experiment
from callbacks.save_results import SaveResults

# from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class ExperimentBadResults(Experiment):
    
    def process_option(self, desc, val):
        cfg = super().process_option(desc, val)

        log_image_path = f"{self.output_path}/images"
        if not os.path.exists(log_image_path):
            os.makedirs(log_image_path)
            
        cfg.net_config.callbacks[0]["args"]["path"] = log_image_path
        
        return cfg

data_cfg: DataConfig = DATA_CONFIG_RGB

net_cfg: NetConfig = NET_CONFIG_IMAGENET_MOBILENET
net_cfg.callbacks += get_save_result_callbacks(frequency="batch")

cfg: Config = Config(
    batch_size=32,
    num_workers=1,
    num_epochs=20,
    data_config=data_cfg,
    net_config=net_cfg,
    log_config=LogConfig(
        log_model="all",
        wandb_logger=False,
    )
)

experiment = ExperimentBadResults("bad_results", cfg)
experiment.run([('test', None)])

