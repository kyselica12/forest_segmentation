import numpy as np 
import matplotlib.pyplot as plt
import tifffile
import os
import skimage 
import sys  

sys.path.append("/home/daniel/Documents/hsi-forest-segmentation/Kyselica/src")


N = 111
image_path = f"/media/daniel/data/sentinel2/2021_seasons/Belgium_summer2021/images/Belgium_image_tile_{N:04d}.tif"
mask_path =  f"/media/daniel/data/sentinel2/2021_seasons/Belgium_summer2021/masks/Belgium_mask_tile_{N:04d}.tif"

image = tifffile.imread(image_path).transpose(1,2,0)
mask = tifffile.imread(mask_path)
mask[mask == 10] = 1
mask[mask != 1] = 0


rgb_image = image[:,:,[3,2,1]]

rgb_image = rgb_image / 10000
rgb_image /= np.max(rgb_image)
print(np.max(rgb_image), np.min(rgb_image))

# plot image and mask in one figure

fig, ax = plt.subplots(1,3, figsize=(10,5))
ax[0].imshow(rgb_image)
ax[1].imshow(mask)

composed_image = skimage.color.label2rgb(mask, rgb_image, colors=['yellow', 'blue'], alpha=0.1)

ax[2].imshow(composed_image)

plt.savefig("composed_image.png")

#print current file path   

print(os.path.abspath(__file__))

import re

# path = os.path.abspath(__file__)
path = re.sub(r'/src/.*', '/src', os.path.abspath(__file__))
print(path) 
# path = os.path.abspath(__file__).split("/")
# path = "/".join(path[:path.index('src')+1])

# print(path)

