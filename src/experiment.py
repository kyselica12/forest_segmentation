from copy import deepcopy
from configs.config import *
from configs.constants import PACKAGE_PATH
from utils import train
from typing import Tuple

#FIXME - not working due to refactoring - changes in train function
#      - need to add arguments for Trainer  
class Experiment:
    
    def __init__(self, name, data_cfg: DataConfig, net_cfg: NetConfig, log_to_wandb=False):
        self.data_cfg = data_cfg
        self.net_cfg = net_cfg
        self.log_to_wandb = log_to_wandb
        self.name = name
        
        self.output_root_dir = f"{PACKAGE_PATH}/results/{self.name}"
        
        if not os.path.exists(self.output_root_dir):
            os.makedirs(self.output_root_dir)
        
        self.output_path = None
    
    
    def process_option(self, desc, val) -> Tuple[DataConfig, NetConfig]:
        self.cfg.log_config.name = desc
        self.output_path = f"{self.output_root_dir}/{desc}"
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        return deepcopy(self.data_cfg), deepcopy(self.net_cfg) 
        
        
    def run(self, options, n_epochs, batch_size, num_workers, callbacks, logger):

        for desc, val in options:
            data_cfg, net_cfg = self.process_option(desc, val)
            
            