import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF



class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        
    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        
        super(Unet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # the down part
        
        prev_feature = in_channels
        for feature in features:
            cnv = DoubleConv(prev_feature, feature)
            self.downs.append(cnv)
            prev_feature = feature            
        
        prev_feature = features[-1]
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=2*feature, # added skip connection
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )        
            self.ups.append(DoubleConv(2*feature, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        self.final = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)
        
        
    
    def forward(self, x):
        
        memory = []
        
        # DOWN
        
        for l in self.downs:
            x = l(x)
            memory.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        # UP
        
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = memory[-(i//2)-1]
            
            # resize the upsapled image if its shape is smaller than the original
            if x.shape != skip_connection.shape:
                x = TF.resize(x, skip_connection.shape[2:], antialias=True) # only use width and height to resize
            
            x = torch.concat([skip_connection, x], dim=1)
            x = self.ups[i+1](x)
            
            
        x = self.final(x)
        
        
        
        return x
        

if __name__ == "__main__":
    
    x = torch.rand((5, 3, 16*20 + 1, 16*20 + 1))
    
    model = Unet(in_channels=3, out_channels=1)
    y = model(x)
    
    print(x.shape)
    print(y.shape)
    
    assert x.shape[2:] == y.shape[2:]
        
        
            
        