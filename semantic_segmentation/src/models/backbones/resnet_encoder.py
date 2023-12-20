from copy import deepcopy
import torch.nn as nn
from torchvision.models.resnet import ResNet
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder

class SentinelResNetEncoder(ResNetEncoder):
    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)
        
        self._in_channels = 13
        self.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def set_in_channels(self, in_channels, pretrained=True):
        pass