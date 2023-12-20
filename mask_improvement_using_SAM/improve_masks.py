import os
import tqdm
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage import feature
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry, SamPredictor
from skimage.filters import gaussian

class SamMaskGenerator:
    """
    A class used to generate modified masks from RGB images using a SAM model.
    ...

    Attributes
    ----------
    sam : object
        An instance of the SAM model.
    mask_generator : object
        An instance of the SamAutomaticMaskGenerator.
    sub_min_surrounding : float
        The minimum surrounding ratio for subtraction.
    sub_min_bkg : float
        The minimum background ratio for subtraction.
    add_max_surrounding : float
        The maximum surrounding ratio for addition.
    add_max_bkg : float
        The maximum background ratio for addition.
    max_area : int, optional
        The maximum area for the mask.
    sigma : int
        The sigma value for the Gaussian function.
    device : str
        The device to run the model on. Can be 'cuda' or 'cpu'.
    """
    
    def __init__(self, model_type,
                 ckpt_path,
                 sub_min_surrounding, 
                 sub_min_bkg, 
                 add_max_surrounding=-1, 
                 add_max_bkg=0, 
                 max_area=None, sigma=2,
                 device='cuda'):

        self.sam = sam_model_registry[model_type](ckpt_path)
        self.sam.to(device)

        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side = 30,
            pred_iou_thresh = 0.85,
            stability_score_thresh= 0.90,
            stability_score_offset= 1,
            box_nms_thresh= 0.7,
            crop_n_layers= 0,
            crop_nms_thresh=0.7,
            crop_overlap_ratio=512/1500,
            crop_n_points_downscale_factor= 1,
            point_grids= None,
            min_mask_region_area= 10
        ) 

        self.sub_min_surrounding = sub_min_surrounding
        self.sub_min_bkg = sub_min_bkg
        self.add_max_surrounding = add_max_surrounding
        self.add_max_bkg = add_max_bkg
        self.max_area = max_area
        self.sigma = sigma
        self.device = device
    
    def process_image(self, img, mask):
        """Generate new mask from RGB image and mask."""
        
        anns = self.generate_annotations(img, mask)
        new_mask = self.improve_mask(anns, mask,
                                     self.sub_min_surrounding,self.sub_min_bkg,
                                     self.add_max_surrounding,self.add_max_bkg,
                                     self.max_area, self.sigma)
        return new_mask
    
    def improve_mask(self,blobs, target_mask, 
                    sub_min_surrounding, 
                    sub_min_bkg, 
                    add_max_surrounding, 
                    add_max_bkg, 
                    max_area=None, sigma=2):
        """Create new mas based on the blobs and the target mask.

        Args:
            blobs : SAM annotations.
            target_mask 
            sub_min_surrounding : Min surrounding ratio for subtraction.
            sub_min_bkg : Min background ratio for subtraction.
            add_max_surrounding: Max surrounding ratio fro addition. Set to -1 to disable.
            add_max_bkg : Max background ratio for addition.
            max_area : Defaults to None -> image area
            sigma : Surrounding parameter . Defaults to 2.

        Returns:
            _type_: _description_
        """
        
        new_mask = target_mask.copy()
        
        for m in blobs:
            surrounding_ratio = self.surrounding_ratio_gauss(m['segmentation'], target_mask, sigma=sigma)
            if surrounding_ratio <= add_max_surrounding:
                if m['non_tree_pixels']/m['area'] <= add_max_bkg and m['non_tree_pixels'] <= max_area:
                    new_mask = np.logical_or(new_mask, m['segmentation'])
            
            elif surrounding_ratio >= sub_min_surrounding:
                if m['non_tree_pixels']/m['area'] >= sub_min_bkg and\
                    max_area is None or m['area'] - m['non_tree_pixels'] <= max_area:
                    new_mask = np.logical_and(new_mask, np.logical_not(m['segmentation']))
        
        return new_mask

    def generate_annotations(self, img, mask):
        """Generate SAM annotations for the image and mask.  """

        masked_image = img.copy()
        masked_image[gaussian(mask, sigma=2) == 0] = 0
        masks = self.mask_generator.generate(masked_image)
        sorted_masks = sorted(masks, reverse=True, key=lambda x: x['segmentation'].sum())
        
        for m in sorted_masks:
            m['non_tree_pixels'] = (m['segmentation'] * np.logical_not(mask)).sum()
        
        return sorted_masks

    def surrounding_ratio_gauss(self, segment, mask, sigma=1):
        """Surrounding ratio in sigma distance from border."""
        inner = gaussian(segment, sigma=sigma) > 0
        outer = gaussian(segment, sigma=sigma*2) > 0
        
        bkg = outer * np.logical_not(segment)
        
        return (bkg * mask).sum() / bkg.sum()


if __name__ == '__main__':

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

    FOLDER_PATH = "/media/daniel/data1/sentinel2/2021_seasons"
    DATA_PATH = f"{FOLDER_PATH}/Belgium_autumn2021"
    OUTPUT_PATH = f"{DATA_PATH}/improved_masks"

    SAM_H_CKPT = '/home/daniel/Documents/work/SAM/models/sam_vit_h_4b8939.pth'
    SAM_L_CKPT = '/home/daniel/Documents/work/SAM/models/sam_vit_l_0b3195.pth'
    SAM_B_CKPT = '/home/daniel/Documents/work/SAM/models/sam_vit_b_01ec64.pth'

    TYPE_H = 'vit_h'
    TYPE_L = 'vit_l'
    TYPE_B = 'vit_b'

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    mask_generator = SamMaskGenerator(
        model_type=TYPE_H,
        ckpt_path=SAM_H_CKPT,
        sub_min_surrounding=0.8,
        sub_min_bkg=0.3,
        add_max_surrounding=-1,
        add_max_bkg=0,
        max_area=512**2 * 0.05,
        sigma=2,
        surrounding_mode='canny',
        device='cuda'
    ) 

    for idx in tqdm.tqdm(range(1,3124)):
        
        image, mask = load_image_mask(idx,DATA_PATH)

        new_mask = mask_generator.process_image(image, mask)

        new_mask = new_mask.astype(np.uint8)
        diff = mask.astype(np.int8) - new_mask
        
        tifffile.imwrite(f'{OUTPUT_PATH}/mask_{idx:04}.tif', new_mask) 
        tifffile.imwrite(f'{OUTPUT_PATH}/difference_{idx:04}.tif', diff) 
