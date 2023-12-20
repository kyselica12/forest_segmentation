'''
Utilities for creating input data for the super-resolution model.
'''

import os
import sys
import tifffile
import numpy as np
import cv2 as cv
from typing import List, Tuple, Iterator

Patch = Tuple[Tuple[int, int], np.ndarray]

def load_series(path: str, i: int, n_images: int=6) -> List[np.ndarray]:
    """Load series with index i from path.

    Args:
        path (str): path to the directory containing the series
        i (int): series index
        n_images (int, optional): Number of images in the series. Defaults to 6.

    Returns:
        List[np.ndarray]: List of RGB images
    """
    images = [load_image(path, i, j) for j in range(n_images)]
    return images

def load_image(path: str, i: int, j: int) -> np.ndarray:
    """Read an .tif Sentinel image and transform it to RGB.

    Args:
        path (str): path to the .tif image
        i (int): series index
        j (int): index of the image in the series

    Returns:
        np.ndarray: RGB image
    """
    img_path = f"{path}/image_{i}_{j}.tif" 
    img = tifffile.imread(img_path)
    img = img.transpose(1,2,0) / 10_000
    img = (img * 255).astype(np.uint8)
    
    return img

def create_patches_iter(images: List[np.ndarray], offset: int, patch_size: int) -> Iterator[Patch]:
    """Iterate over image patches.

    Args:
        images (List[np.ndarray]): Series of images
        offset (int): Offset of the patches (overlap)
        patch_size (int): Size of the patches

    Yields:
        (i,j), patch_image : i and j position of the patch in the original image
    """
    dx = patch_size - offset
    dy = patch_size - offset

    for i, x in enumerate(range(0, 512-offset//2, dx)):
        for j, y in enumerate(range(0, 512-offset//2, dy)):
            tile = []
            for img in images:
                one_tile = np.zeros((patch_size, patch_size, 3))
                data = img[x:x+patch_size, y:y+patch_size]
                one_tile[:data.shape[0], :data.shape[1]] = data
                tile.append(one_tile)
                
            assert len(tile) == 6

            tile = np.concatenate(tile)
            yield (i,j), tile

def get_patches(images: List[np.ndarray], offset: int,size: int) -> List[Patch]:
    """List of patches from the series of images.

    Args:
        images (List[np.ndarray]): Series of images
        offset (int): Offset of the patches (overlap)
        size (int): Size of the patches

    Returns:
        List[Patch]: Patch = (i,j), patch_image : i and j position of the patch in the original image
    """
    tiles = [((i,j), p) for (i,j), p in create_patches_iter(images, offset, size)]
    return tiles

def save_sub_images(patches: Iterator[Patch] | List[Patch], path: str, desc: str='')-> None:
    """Save pathces in the format <path>/<desc>_<i>_<j>.png

    Args:
        patches (Iterator[Patch] | List[Patch]): 
        path (str): path to the output directory
        desc (str, optional): Additional description. Defaults to ''.
    """
    output_path = f"{path}/{desc}"
    os.makedirs(output_path, exist_ok=True)
    for (x,y), tile in patches:
        cv.imwrite(f'{output_path}/{desc}_{x}_{y}.png', tile[:,:,[2,1,0]])
    
if __name__ == "__main__":    
    PATH = "/media/daniel/data1/sentinel2/Wallonia_for_SatlasSR/spring_2018/sentinel_images"
    # OUTPUT_PATH = "/tmp/data/"
    OUTPUT_PATH = "./images"

    N = 6
    SIZE = 32   
    OFFSET = 8 
    SERIES_IDX = 0
    
    images = load_series(PATH, SERIES_IDX, N)
    patches = get_patches(images, OFFSET, SIZE)
    save_sub_images(patches, OUTPUT_PATH, f"image_{SERIES_IDX}")