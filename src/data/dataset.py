import re
from torch.utils.data import Dataset
import torch
import os
import tifffile
import tqdm
import numpy as np

from configs.constants import ALL_BANDS_LIST, IGNORE_INDDEX, MASK_DIFFERENCE_INDEX

class SentinelDataset(Dataset):
    
    
    def __init__(self, image_list, mask_list,
            bands=ALL_BANDS_LIST, 
            label_mappings=None, 
            transforms=None, 
            scale=10000,
            mean=None,
            std=None,
            mode='basic',
            use_level_C1=False,
            improved_mask=False):
        
        self.image_list = image_list
        self.mask_list = mask_list
        self.label_mappings = label_mappings
        self.transforms = transforms 
        self.use_bands = bands
        self.use_level_C1 = use_level_C1

        self.mean = mean
        self.std = std
        self.scale = scale

        self.improved_mask = improved_mask
            
    def _normalize(self, image):
        
        mask = np.abs(np.sign(image)) # do not normalize zero pixels
        image = (image - self.mean) / self.std / self.scale
        
        return image * mask
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        
        img_path = self.image_list[idx]
        mask_path = self.mask_list[idx]

        
        image = tifffile.imread(img_path).transpose(1,2,0)
        image = self._normalize(image)
        image = image[:, :, self.use_bands]
        image[np.isnan(image)] = 0

        empty = image.sum(axis=2) == 0

        mask = tifffile.imread(mask_path)
        
        if self.improved_mask:
            folder_path, name = os.path.split(img_path)
            
            res= re.findall(r'(\d+)', name)  
            image_idx = int(res[-1])
            
            folder_path = os.path.split(folder_path)[0]
            
            # improved_mask = tifffile.imread(f'{folder_path}/improved_mask/mask{image_idx:04}.tif')
            difference = tifffile.imread(f'{folder_path}/improved_masks/difference_{image_idx:04}.tif')

            mask[difference == 1] = MASK_DIFFERENCE_INDEX

        mask[empty] = IGNORE_INDDEX
                
        if self.label_mappings is not None:
            for key, value in self.label_mappings.items():
                mask[mask == key] = value
            
        if self.transforms is not None:
            aug = self.transforms(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        
        if self.use_level_C1:
            image = np.insert(image, 10, 0, axis=0)
                        
        return image, mask
    
    
if __name__ == "__main__":
    
    from data.utils import load_dataset_csv, split_data
    
    # split_data("/media/daniel/data/sentinel2/Belgium_summer2022/", output_path='./')
    
    X_train, X_val, y_train, y_val = load_dataset_csv('./')
    
    ds = SentinelDataset(X_train, y_train, mode='train')
    
    mean, std = ds.compute_mean_std()
    
    s = ", ".join([str(x) for x in mean])
    print("MEAN: [{}]".format(s))
    s = ", ".join([str(x) for x in std])
    print("STD: [{}]".format(s))
