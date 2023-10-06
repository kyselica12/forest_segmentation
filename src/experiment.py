from copy import deepcopy
from configs.config import *
from configs.constants import PACKAGE_PATH
from utils import train
from typing import Tuple

from models.image_segmentation_module import ImageSegmentationModule
from data.data_processor import DataProcessor
from utils import train, get_wabdb_logger

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
        
        self.module = None
        self.data_processor = None
    
    
    def process_option(self, desc, val) -> Tuple[DataConfig, NetConfig]:
        
        return deepcopy(self.data_cfg), deepcopy(self.net_cfg) 
    
    def get_callbacks(self, desc, val):
        return []
        
    def get_module(self, desc, val, net_cfg):
        return ImageSegmentationModule(**net_cfg.__dict__)
    
    def get_data_processor(self, desc, val, data_cfg):
        return DataProcessor(data_cfg)
    
    def run(self, options, n_epochs, batch_size, num_workers):

        for desc, val in options:
            data_cfg, net_cfg = self.process_option(desc, val)

            module = self.get_module(desc, val, net_cfg)
            data_processor = self.get_data_processor(desc, val, data_cfg)
            
            logger = None
            if self.log_to_wandb:
                logger = get_wabdb_logger(self.name, desc)
            
            callbacks = self.get_callbacks(desc, val)

            train(module, data_processor, n_epochs, batch_size, num_workers, callbacks, logger)

            
            