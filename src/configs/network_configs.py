from copy import deepcopy
from config import *

from callbacks.save_results import SaveResults



DEFAULT_NET_CONFIG = NetConfig(
    architecture=NetworkArchitectures.UNET,
    config={
        'encoder_name': 'timm-mobilenetv3_small_minimal_100',
        'encoder_weights': 'imagenet',
    },callbacks = []
)

NET_CONFIG_IMAGENET_MOBILENET = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_IMAGENET_MOBILENET.config = {
    'encoder_name': 'timm-mobilenetv3_small_minimal_100',
    'encoder_weights': 'imagenet',
}

NET_CONFIG_IMAGENET_RESNET18 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_IMAGENET_RESNET18.config = {
    'encoder_name': 'resnet18',
    'encoder_weights': 'imagenet',
}

NET_CONFIG_IMAGENET_RESNET50 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_IMAGENET_RESNET50.config = {
    'encoder_name': 'resnet50',
    'encoder_weights': 'imagenet',
}

NET_CONFIG_S2_RESNET50 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_S2_RESNET50.config = {
    'encoder_name': CustomNets.RESNET50,
    'encoder_weights': CustomNetWeights.RESNET50,
}

NET_CONFIG_S2_RESNET18 = deepcopy(DEFAULT_NET_CONFIG)
NET_CONFIG_S2_RESNET18.config = {
    'encoder_name': CustomNets.RESNET18,
    'encoder_weights': CustomNetWeights.RESNET18,
}

def get_save_result_callbacks(num=3, path='', frequency='epoch', monitor='loss', mode='max', size=4):
    
    return [{
        "class": SaveResults,
        "args": {
            "num": num,
            "path": path, 
            "frequency": frequency,
            "monitor": monitor,
            "mode": mode, 
            "size": size  
        }
    }]
