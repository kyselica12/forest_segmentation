import numpy as np
from skimage import feature
from skimage.filters import gaussian
import tifffile


SAM_H_CKPT = '/home/daniel/Documents/work/SAM/models/sam_vit_h_4b8939.pth'
SAM_L_CKPT = '/home/daniel/Documents/work/SAM/models/sam_vit_l_0b3195.pth'
SAM_B_CKPT = '/home/daniel/Documents/work/SAM/models/sam_vit_b_01ec64.pth'

TYPE_H = 'vit_h'
TYPE_L = 'vit_l'
TYPE_B = 'vit_b'

from segment_anything import sam_model_registry, SamPredictor

def get_SAM(type='H'):
    sam = None
    if type == 'H':
        sam = sam_model_registry[TYPE_H](SAM_H_CKPT)
        sam.cuda()
    elif type == 'L':
        sam = sam_model_registry[TYPE_L](SAM_L_CKPT)
        sam.cuda()
    else:
        sam = sam_model_registry[TYPE_B](SAM_B_CKPT)
        sam.cuda()
        
    return sam

def generate_sam_annotations(mask_generator, img, mask):
    masks = mask_generator.generate(img)
    sorted_masks = sorted(masks, reverse=True, key=lambda x: x['segmentation'].sum())
    
    for m in sorted_masks:
        m['non_tree_pixels'] = (m['segmentation'] * np.logical_not(mask)).sum()
    
    return sorted_masks

def surrounding_ratio_canny(segment, mask):    
    edges = feature.canny(segment, sigma=3)
    return (edges * mask).sum() / edges.sum()

def surrounding_ratio_gauss(segment, mask, sigma=1):
    inner = gaussian(segment, sigma=sigma) > 0
    outer = gaussian(segment, sigma=sigma*2) > 0
    
    bkg = outer * np.logical_not(segment)
    
    # dif = blures * np.logical_not(segment)
    
    return (bkg * mask).sum() / bkg.sum()

def surrounding_ratio(segment, mask, mode='canny'):
    if mode == 'canny':
        return surrounding_ratio_canny(segment, mask)
    elif mode == 'blur':
        return surrounding_ratio_gauss(segment, mask)
    else:
        raise Exception(f'Unknown mode: {mode}')

def load_image_mask(idx, PATH):
    image_path = f'{PATH}/images/Belgium_image_tile_{idx:04}.tif'
    mask_path = f'{PATH}/masks/Belgium_mask_tile_{idx:04}.tif'
    
    img = tifffile.imread(image_path)
    img = img.transpose(1, 2, 0)
    img = img[:, :, [3, 2, 1]]
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    mask = tifffile.imread(mask_path)
    mask = mask == 10
    mask_image = np.zeros((mask.shape)+(3,))
    mask_image[mask] = np.array([0, 255, 0])

    img_T = img.copy()
    img_T[mask != 1] = np.array([0, 0, 0])

    img_F = img.copy()
    img_F[mask] = np.array([0, 0, 0])
    
    return img, mask


def improve_mask(masks, target_mask, 
                 add_max_non_tree_ratio, 
                 sub_min_non_tree_ratio, 
                 add_max_tree_surrounding_ratio, 
                 sub_min_tree_surrounding_ratio, 
                 max_area, sigma):
    
    new_mask = target_mask.copy()
    
    for m in masks:
        surrounding_ratio = surrounding_ratio_gauss(m['segmentation'], target_mask, sigma=sigma)
        if surrounding_ratio <= add_max_tree_surrounding_ratio:
            if m['non_tree_pixels']/m['area'] <= add_max_non_tree_ratio and m['non_tree_pixels'] <= max_area:
                new_mask = np.logical_or(new_mask, m['segmentation'])
        
        elif surrounding_ratio >= sub_min_tree_surrounding_ratio:
            if m['non_tree_pixels']/m['area'] >= sub_min_non_tree_ratio and m['area'] - m['non_tree_pixels'] <= max_area:
                new_mask = np.logical_and(new_mask, np.logical_not(m['segmentation']))
    
    return new_mask

def improve_mask_subtraction(masks, target_mask, 
                 surrounding_ratio, 
                 non_tree_ratio, 
                 max_area, sigma):
    
    new_mask = target_mask.copy()
    
    for m in masks:
        
        surrounding = surrounding_ratio_gauss(m['segmentation'], target_mask, sigma=sigma)
        
        if surrounding >= surrounding_ratio:
            if m['non_tree_pixels']/m['area'] >= non_tree_ratio and m['area'] - m['non_tree_pixels'] <= max_area:
                new_mask = np.logical_and(new_mask, np.logical_not(m['segmentation']))
    
    return new_mask
