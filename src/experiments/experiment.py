from copy import deepcopy
from config import *
from utils import train

class Experiment:
    
    def __init__(self, name, cfg: Config):
        self.cfg = cfg
        self.name = name
        
        self.output_root_dir = f"{PACKAGE_PATH}/results/{self.name}"
        
        if not os.path.exists(self.output_root_dir):
            os.makedirs(self.output_root_dir)
        
        self.output_path = None
    
    
    def process_option(self, desc, val) -> Config:
        self.cfg.log_config.name = desc
        self.output_path = f"{self.output_root_dir}/{desc}"
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        return deepcopy(self.cfg) 
        
        
    def run(self, options):

        for desc, val in options:
            cfg = self.process_option(desc, val)
            train(cfg)