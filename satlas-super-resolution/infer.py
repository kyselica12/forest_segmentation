import os
import random
import glob
import math
import re
import shutil
import tqdm
from typing import List, Tuple, Iterator

import torch
import cv2
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from basicsr.archs.rrdbnet_arch import RRDBNet

totensor = torchvision.transforms.ToTensor()

Patch = Tuple[Tuple[int, int], np.ndarray]

def infer(s2_data: np.ndarray, n_s2_images: int, model: RRDBNet, device: str) -> torch.Tensor:
    """Model inference.

    Args:
        s2_data (np.ndarray): input_data
        n_s2_images (int): number of S2 images used in patch
        model (RRDBNet) 
        device (str): Default: 'cuda'

    Returns:
        torch.Tensor: model output
    """
    # Reshape to be Tx32x32x3.
    s2_chunks = np.reshape(s2_data, (-1, 32, 32, 3))

    # Iterate through the 32x32 chunks at each timestep, separating them into "good" (valid)
    # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
    goods, bads = [], []
    for i,ts in enumerate(s2_chunks):
        if [0, 0, 0] in ts:
            bads.append(i)
        else:
            goods.append(i)

    # Pick {n_s2_images} random indices of s2 images to use. Skip ones that are partially black.
    if len(goods) >= n_s2_images:
        rand_indices = random.sample(goods, n_s2_images)
    else:
        need = n_s2_images - len(goods)
        rand_indices = goods + random.sample(bads, need)

    s2_chunks = [s2_chunks[i] for i in rand_indices]
    s2_chunks = np.array(s2_chunks)

    # Convert to torch tensor.
    s2_chunks = [totensor(img) for img in s2_chunks]
    s2_tensor = torch.cat(s2_chunks).unsqueeze(0).to(device)

    # Feed input of shape [batch, n_s2_images * channels, 32, 32] through model.
    output = model(s2_tensor)
    return output

def stitch(patches: List[Patch] | Iterator[Patch],
           scale: int=4,
           offset: int=10,
           src_image_size: int=512,
           patch_size: int=32) -> np.ndarray:
    """Stitch pathces together into a single image.

    Args:
        patches (List[Patch] | Iterator[Patch])
        scale (int, optional): Scaling factor. Defaults to 4.
        offset (int, optional): Offset (overlay) of the pathces. Defaults to 10.
        src_image_size (int, optional): Size of input image(s). Defaults to 512.
        patch_size (int, optional): Patch size. Defaults to 32.

    Returns:
        np.ndarray: Super-resolved image 
    """

    img_size = src_image_size * scale
    chunk_size = patch_size * scale
    n_images = math.ceil((src_image_size) / (patch_size - offset))
    target_offset = offset * scale

    result = np.zeros((img_size, img_size, 3))

    margin = target_offset // 2
    # mask out the borders
    border_mask = np.ones((chunk_size, chunk_size, 3))
    border_mask[:margin, :, :] = 0 # top strip
    border_mask[-margin:, :, :] = 0 # bottom strip
    border_mask[:, :margin, :] = 0 # left strip
    border_mask[:, -margin:, :] = 0 # right strip

    for (i,j), img in patches:
        mask = border_mask.copy()
        if i == 0: # top row
            mask[:margin, margin:-margin, :] = 1
        if i == n_images - 1: # bottom row
            mask[-margin:, margin:-margin, :] = 1
        if j == 0: # left column
            mask[margin:-margin, :margin, :] = 1
        if j == n_images - 1: # right column
            mask[:, -margin:, :] = 1
        
        if i == 0 and j == 0: # top left corner
            mask[:margin, :margin, :] = 1
        if i == 0 and j == n_images - 1: # top right corner
            mask[:margin, -margin:, :] = 1
        if i == n_images - 1 and j == 0: # bottom left corner
            mask[-margin:, :margin, :] = 1
        if i == n_images - 1 and j == n_images - 1: # bottom right corner
            mask[-margin:, -margin:, :] = 1

        img = img * mask 


        dx = (patch_size - offset) * scale

        x_size = min(i*dx + chunk_size, img_size) - i*dx
        y_size = min(j*dx + chunk_size, img_size) - j*dx

        result[i*dx:min(i*dx + chunk_size, img_size), j*dx:min(j*dx+chunk_size, img_size), :] += img[:x_size, :y_size, :]

    return result
    
def sr_image(img: np.ndarray, n_s2_images: int, model: RRDBNet, device: str) -> np.ndarray:
    """Creates a super-resolved image patch.

    Args:
        img (np.ndarray): image patch
        n_s2_images (int): Number of sentinel images used for patch
        model (RRDBNet)
        device (str)

    Returns:
        np.ndarray: Super-resolved image patch
    """
    img = (img / 255).astype(np.float32)

    output = infer(img, n_s2_images, model, device)
    output = output.squeeze().cpu().detach().numpy()
    output = np.transpose(output, (1, 2, 0)) 

    # output = output + np.abs(np.min(output, axis=(0,1)))
    # output = output / np.max(output, axis=(0,1))

    output = output * 255
    output[output < 0] = 0
    output[output > 255] = 255
    output = output.astype(np.uint8)

    return output

def save_pathces(patches: List[Patch] | Iterator[Patch], path: str) -> None:
    """Save pathces in the format <path>/image_<i>_<j>.png

    Args:
        patches (List[Patch] | Iterator[Patch]): Patches to save
        path (str)
    """
    os.makedirs(path, exist_ok=True)
    for (i,j), img in patches:
        cv2.imwrite(f"{path}/image_{i}_{j}.png", img[:,:,[2,1,0]])

def load_patches_iter(path: str) -> Iterator[Patch]:
    """Loads patches from a directory.

    patches saved in format:
        image_<i>_<j>.png

    Args:
        path (str) 

    Raises:
        ValueError: If image name cannot be parsed for i and j coordinates.

    Yields:
        Iterator[Patch]: Patches
    """
    pngs = glob.glob(path + "/*.png")
    pngs = sorted(pngs) # -> important to return images in order of i,j

    for png_path in pngs:
        name = os.path.split(png_path)[-1]
        try:
            i, j = tuple(map(int, re.findall(r"\d+", name)))
        except:
            raise ValueError(f"Could not parse {name} for x and y coordinates.")

        img = cv2.imread(png_path)[:, :, [2,1,0]]

        yield (i,j), img

def load_pathces(path: str) -> List[Patch]:
    """Loads patches from a directory into a list.

    Args:
        path (str)

    Returns:
        List[Patch]: List of patches
    """
    return list(load_patches_iter(path))

def load_model(weights_path:str, device:str, n_s2_images=6) -> RRDBNet:
    """Load weights into RRDBNet model.

    Args:
        weights_path (str): path to the .pth file containing the weights
        device (str)
        n_s2_images (int, optional): Size of the models input . Defaults to 6.

    Returns:
        RRDBNet: _description_
    """
    model = RRDBNet(num_in_ch=n_s2_images*3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict['params_ema'])
    model.eval()

    return model

def process_pathces_iter(pathces_iter: List[Patch] | Iterator[Patch],
                         model: RRDBNet,
                         device: str,
                         n_s2_images: int=6) -> Iterator[Patch]:
    """Create super-resolved patches from input patches.

    Args:
        pathces_iter (List[Patch] | Iterator[Patch])
        model (RRDBNet)
        device (str)
        n_s2_images (int, optional): Model input series size. Defaults to 6.

    Returns:
        _type_: Iterator[Patch]
    """
    return map(lambda x: (x[0] ,sr_image(x[1], n_s2_images, model, device)), pathces_iter)

def process_pathces(patches: List[Patch]|Iterator[Patch],
                    model: RRDBNet,
                    device: str,
                    n_s2_images: int=6)-> List[Patch]:
    """Create super-resolved patches from input patches.

    Args:
        patches (List[Patch] | Iterator[Patch])
        model (RRDBNet)
        device (str)
        n_s2_images (int, optional): Model input series size. Defaults to 6.

    Returns:
        List[Patch]: list of super-resolved patches
    """
    return list(process_pathces_iter(patches, model, device, n_s2_images))

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--data_dir', type=str, help="Path to the directory containing images.")
    # parser.add_argument('-w', '--weights_path', type=str, default="weights/esrgan_orig_6S2.pth", help="Path to the model weights.")
    # parser.add_argument('--n_s2_images', type=int, default=6, help="Number of Sentinel-2 images as input, must correlate to correct model weights.")
    # parser.add_argument('--save_path', type=str, default="outputs", help="Directory where generated outputs will be saved.")
    # parser.add_argument('--stitch', action='store_true', help="If running on 16x16 grid of Sentinel-2 images, option to stitch together whole image.")
    # args = parser.parse_args()

    device: str = 'cuda'

    # delete directory with contents

    # shutil.rmtree("/tmp/outputs")
        

   
    patches_dir: str = "./images/image_0" 
    sr_patches_dir: str = "./sr_patches"
    n_s2_images: int = 6
    weights_path: str = f"./weights/esrgan_orig_{n_s2_images}S2.pth"

    model: RRDBNet = load_model(weights_path, device)
    orig_patches_iter = load_patches_iter(patches_dir)
    orig_pathces = load_pathces(patches_dir)
    sr_patches_iter = process_pathces_iter(orig_patches_iter, model, device, n_s2_images)
    sr_patches = list(sr_patches_iter)
    save_pathces(sr_patches, sr_patches_dir)

    # sr_patches = load_patches_iter(sr_patches_dir)
    # sr_patches = process_pathces(orig_patches_iter, model, device, n_s2_images)

    sr_image = stitch(sr_patches,
                    scale=4,
                    grid_size=16,
                    offset=8,
                    src_image_size=512)
    
    cv2.imwrite(f"sr_image_{n_s2_images}_test.png", sr_image[:,:,[2,1,0]])


