import argparse
import glob
import os
from typing import List, Iterator

import numpy as np
import tifffile
import tqdm
import cv2 as cv
from basicsr.archs.rrdbnet_arch import RRDBNet

from create_input import load_series, get_patches, Patch
from infer import load_model, process_pathces_iter, stitch

test = np.array([1,2,3,5])

# crate parser with description
parser = argparse.ArgumentParser(
    prog='SR with SATLAS',
    description=\
'''
Script for creating Super-Resolution images of Sentinel-2 images using SATLAS-SR.
Resulting image is in 4x resolution of the input image (10m/px -> 2.5m/px).

Source: https://github.com/allenai/satlas-super-resolution

'''
)
parser.add_argument('-i', '--input-dir', type=str, default=".",
                    help="Path to the directory containing series of images (image_x_y.tif).")
parser.add_argument('-o', '--output-dir', type=str, default=".",
                    help="Path to the directory where the output images will be saved.")
parser.add_argument('-n', '--n-s2-images', type=int, default=6,
                    help="Number of Sentinel-2 images in the series for SATLAS-SR.")
parser.add_argument('-w', '--weights-path', type=str, default="./weights",
                    help="Path to the model weights .pth file.")
parser.add_argument('--output-size', type=int, default=512, help="Size of the output image / images (if the SR image is larger than output_size).")
parser.add_argument('--patch-size', type=int, default=32,help="Size of the intermediate patches.")
parser.add_argument('--patch-offset', type=int, default=10, help="Offset of the intermediate patches to avoid border effects on final image.")
parser.add_argument('--original-size', type=int, default=512, help="Size of the original image.")
parser.add_argument('--device', type=str, default='cuda', help="Device to use for inference.")
parser.add_argument('--offset', type=int, default=0, help="Offset of the series to start from.")
parser.add_argument('--end', type=int, default=None, help="Index of the last series to process.")

args = parser.parse_args()

if __name__ == "__main__":

    INPUT_PATH = args.input_dir
    OUTPUT_PATH = args.output_dir
    OUTPUT_SIZE = args.output_size
    START = args.offset + 1 # start indexing from 1
    END = args.end 
    
    N_S2_IMAGES = args.n_s2_images
    PATCH_OFFSET = args.patch_offset
    PATCH_SIZE = args.patch_size
    ORIG_SIZE = args.original_size
    
    WEIGHTS_PATH = args.weights_path #f"{args.weights_path}/esrgan_orig_{N_S2_IMAGES}S2.pth"
    DEVICE = args.device

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    model: RRDBNet = load_model(WEIGHTS_PATH, DEVICE, N_S2_IMAGES)

    indices = sorted([int(os.path.split(p)[-1].split('_')[1]) for p in glob.glob(f"{INPUT_PATH}/image_*_0.tif")])
    END = END if END is not None else len(indices)
    for idx in tqdm.tqdm(range(START, END + 1)):
        images = load_series(INPUT_PATH, idx, N_S2_IMAGES)
        patches_iter = get_patches(images, PATCH_OFFSET, PATCH_SIZE)

        sr_patches_iter = process_pathces_iter(patches_iter, model, DEVICE, N_S2_IMAGES)

        sr_image = stitch(sr_patches_iter,
                        scale=4,
                        offset=PATCH_OFFSET,
                        src_image_size=ORIG_SIZE)
        
        if sr_image.shape[0] > OUTPUT_SIZE:
            for i, x in enumerate(range(0, sr_image.shape[0], OUTPUT_SIZE)):
                for j, y in enumerate(range(0, sr_image.shape[1], OUTPUT_SIZE)):
                    res = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3))
                    x2 = min(x+OUTPUT_SIZE, sr_image.shape[0])
                    y2 = min(y+OUTPUT_SIZE, sr_image.shape[1])
                    # img = sr_image[x:x2, y:y2]
                    res[:x2-x, :y2-y] = sr_image[x:x2, y:y2]
                    res = res.transpose(2,0,1) # transpose to CxHxW format
                    tifffile.imwrite(f"{OUTPUT_PATH}/{idx}_{i}_{j}.tif", res)
        else:    
            sr_image = sr_image.transpose(2,0,1) # transpose to CxHxW format
            tifffile.imwrite(f"{OUTPUT_PATH}/{idx}.tif", sr_image)



