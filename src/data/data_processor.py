import glob
import os
import re
import json
import tqdm
import hashlib
import tifffile
import tqdm

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader


from data.dataset import SentinelDataset
from configs.constants import ALL_BANDS_LIST, ALL_CLASSES_SET 
from configs.config import ESAWorldCover, Sentinel2Bands, DataConfig





class DataProcessor:
    """Class for managing the data processing pipeline.
    
    Attributes:
        DataConfig: DataConfig object containing the configuration of the data processing pipeline.
        widtd, height : width and height of the image 
        bands : list of Sentinel 2 bands to use
        classes : list of classes to use
        stabilization_scale_factor : scale factor to use for numerical stability default 10_000
        batch_size : batch size to use for training
        num_workers : number of workers to use for training
        val_size : validation proportion
        random_state : random state to use for splitting the data
        output_path : path to save the output or load existing data
        train_path : path to the training data
        val_path : path to the validation data
        grid_path : path to the grid json file
        load : whether to load existing data
        compute_mean_std : whether to use computed mean and std for normalization
        use_level_C1 : whether the model is pretrained on Sentinel2 level C1 data
    """
    
    def __init__(self, cfg: DataConfig):

        self.cfg = cfg
        
        self.load = cfg.load
        self.train_path = cfg.train_path
        self.val_path = cfg.val_path
        self.grid_path = cfg.grid_path
        self.output_path = cfg.output_path
        self.val_size = cfg.val_size
        self.random_state = cfg.random_state
        
        self.N = 0 
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_set = None
        self.val_set = None
        
        self.non_zero_ration = cfg.non_zero_ratio
        
        self.mean = np.zeros((len(ALL_BANDS_LIST),))
        self.std = np.ones((len(ALL_BANDS_LIST),))
        
        self.bands = cfg.bands
        self.classes = cfg.classes
        self.scale = cfg.stabilization_scale_factor
        self.width = cfg.width
        self.height = cfg.height

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        
        self.use_level_C1 = cfg.use_level_C1
        self.compute_mean_std = cfg.compute_mean_std
        self.improved_mask = cfg.improved_mask

        hash_string = repr((  self.scale,
                              self.width, self.height, 
                              self.train_path, self.val_path, self.grid_path, 
                              self.val_size, self.random_state, self.non_zero_ration))
        self.hash = hashlib.md5(hash_string.encode('utf-8')).hexdigest()
        print(f"Dataset hash: {self.hash}")

        self.output_path = f"{self.output_path}/{self.hash}"
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        
        self.prepare_datasets()
        
    def prepare_datasets(self):
        """Three main steps:
            1. Load data from files
            2. Split data into train and validation sets
            3. Save data to csv files
            4. Load or compute mean and std
        """

        if self.load and os.path.exists(f"{self.output_path}/train.csv"):
            self.load_csv_dataset()
        else:
            self.create_datasets()
            self.save_csv_dataset()
        
        if self.compute_mean_std:
            if os.path.exists(f"{self.output_path}/mean_std.txt"):
                self.mean, self.std = self._load_mean_std(self.output_path)
            else:
                self.mean, self.std = self._compute_mean_std(self.X_train, self.y_train, self.scale, self.width, self.height)
                self._save_mean_std(self.output_path)

    def save_csv_dataset(self):
        """Save dataset to .csv files
        """
        train_csv = pd.DataFrame({"image": self.X_train, "mask": self.y_train})
        test_csv = pd.DataFrame({"image": self.X_val, "mask": self.y_val})
        
        if self.output_path is not None:
            train_csv.to_csv(f"{self.output_path}/train.csv", index=False)
            test_csv.to_csv(f"{self.output_path}/test.csv", index=False)
    
    def create_datasets(self):
        """Load and split data into train and validation sets
        """
        
        train_images, train_masks = self._load(self.train_path)

        self.N = len(train_images) if self.grid_path is None else self._get_grid_size(self.grid_path)
         
        train_indices, val_indices = self._split_train_val_indices(self.N, val_size=self.val_size, random_state=self.random_state)
        
        
        if self.val_path is not None:
            val_images, val_masks = self._load(self.val_path)
            (val_images, val_masks), _ = self._filter_by_indices(val_images, val_masks, val_indices)
            (train_images, train_masks), _ = self._filter_by_indices(train_images, train_masks, train_indices)
        else:
            (train_images, train_masks), (val_images, val_masks) = self._filter_by_indices(train_images, train_masks, train_indices)
        
        self.X_train = train_images
        self.y_train = train_masks
        self.X_val = val_images
        self.y_val = val_masks
        
    def load_csv_dataset(self):
        """
        Load dataset from .csv files
        """
        train_csv = pd.read_csv(f"{self.output_path}/train.csv")
        test_csv = pd.read_csv(f"{self.output_path}/test.csv")
        
        self.X_train = train_csv["image"].values
        self.y_train = train_csv["mask"].values
        
        self.X_val = test_csv["image"].values
        self.y_val = test_csv["mask"].values
        
    def _load(self, path):
        """Read images and masks paths from folder recursively

        Args:
            path (str): path to the root folder

        Returns:
            tuple(list, list): tuple : list of image paths, list of mask paths
        """
        
        image_list = []
        mask_list = []
        if os.path.exists(f"{path}/images"):
            return self._load_data_folder(path) 

        for filename in sorted(glob.glob(f"{path}/*")):
            if os.path.isdir(filename):
                images, masks = self._load(filename)
            
                image_list.extend(images)
                mask_list.extend(masks)
        
        return image_list, mask_list
        
    def _get_id_from_filename(self, path):
        """Extract number of the image from the filename

        Args:
            path (str): path to the image

        Raises:
            ValueError: file without number

        Returns:
            int: file number
        """
        
        name = os.path.split(path)[-1]
        res = re.findall(r"\d+", name)
        
        if len(res) == 0:
            raise ValueError("Invalid filename")
        
        return int(res[-1])
        
    def _load_data_folder(self, folder):
        """Load image and masks paths from one folder

        Args:
            folder (str): path to the folder

        Returns:
            tuple[list[int], list[int]]: train and validation indices
        """
        
        all_images = sorted(glob.glob(f"{folder}/images/*.tif"))
        all_masks = sorted(glob.glob(f"{folder}/masks/*.tif"))

        image_list = []
        mask_list = []

        pb = tqdm.tqdm(total=len(all_images), desc=f'Loading: {folder}')

        for img, mask in zip(all_images, all_masks):
            if self._image_zero_ratio(img) <= (1-self.non_zero_ration):
                image_list.append(img)
                mask_list.append(mask)
            pb.update(1)
        pb.close()

        print(f"Loaded: {len(image_list)}, discarded: {len(all_images)-len(image_list)}")
        
        return image_list, mask_list
    
    def _split_train_val_indices(self, N, val_size=0.2, random_state=42):
        """Split indices into train and validation sets

        Args:
            N (int): size of the dataset
            val_size (float, optional): validation split . Defaults to 0.2.
            random_state (int, optional)

        Returns:
            _type_: _description_
        """
        prev_seed = np.random.get_state()
        np.random.seed(random_state)
        indices = np.random.permutation(N)
        np.random.set_state(prev_seed)
        
        split = int((1-val_size)*N)
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        return train_indices, val_indices
    
    def _filter_by_indices(self, image_list, mask_list, indices):
        """Filter images and masks by indices

        Args:
            image_list (list): list of image paths
            mask_list (list): list of mask paths
            indices (list): indices to filter by

        Returns:
            tuple: (images, masks with number in indices), (images, masks without number in indicesl)
        """
        
        in_images, in_masks = [], []
        out_images, out_masks = [], []
        
        for img, mask in zip(image_list, mask_list):
            img_id = self._get_id_from_filename(img)
            if img_id in indices:
                in_images.append(img)
                in_masks.append(mask)
            else:
                out_images.append(img)
                out_masks.append(mask)
        
        
        return (in_images, in_masks), (out_images, out_masks)
    
    def _get_grid_size(self, grid_json_path):
        """Read the number of grid cells from the grid json file

        Args:
            grid_json_path (str): path to the grid json file

        Returns:
            int: number of grid cells
        """
        
        with open(grid_json_path, 'r') as f:
            grid = json.load(f)
        
        return len(grid["features"])
    
    def _load_mean_std(self, path):
        """Load mean and std from numpy file

        Args:
            path (str): path to the file

        Returns:
            tuple: mean, std
        """
        mean, std = np.loadtxt(f"{path}/mean_std.txt")
        return mean, std

    def _save_mean_std(self, path):
        """Save mean and std to numpy file

        Args:
            path (str): path to the file
        """
        np.savetxt(f"{path}/mean_std.txt", [self.mean, self.std])
        
    def _compute_mean_std(self, X_path_list, y_path_list, scale, width, height, batch_size=32):
        """Compute mean and std from training set

        Args:
            X_path_list (list): list of image paths
            y_path_list (list): list of mask paths
            scale (int): scale factor for numerical stability
            width (int): image width
            height (height): image height
            batch_size (int, optional): batch size for dataloader. Defaults to 32.

        Returns:
            tuple[array, array]: mean, std (arrays)
        """
        
        n_channels = len(ALL_BANDS_LIST)
        transforms = self._get_transformations(width, height, mode='basic')
        dataset = SentinelDataset(X_path_list, y_path_list, 
                                  bands=ALL_BANDS_LIST, 
                                  transforms=transforms,
                                  mean=np.zeros((n_channels)), std=np.ones((n_channels)),
                                  scale=self.scale)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)    
        
        psum = np.zeros((n_channels,))
        psum_sq = np.zeros((n_channels,))
        
        n = 0
        
        for x, _ in tqdm.tqdm(loader, desc='Computing mean and std from training set'):

            x = x.detach().cpu().numpy()
            
            masked_x = np.ma.masked_equal(x, 0)
            
            masked_x = masked_x / scale # for stability
            im_sum = masked_x.sum(axis=(0,2,3))
            im_sum_sq = (masked_x ** 2).sum(axis=(0,2,3))            
            
            if np.sum(im_sum.mask) > 0:
                print("Masked pixels: {}".format(np.sum(im_sum.mask)))
                continue
            
            n += x.shape[0]
            
            psum += im_sum
            psum_sq += im_sum_sq
        
        N = width * height * n
        mean = psum / N
        std = np.sqrt(psum_sq / N - mean ** 2)
            
        return mean, std
    
    def _get_transformations(self, width, height, mode='basic'):
        """Create list of transformations

        Args:
            width (int): image width
            height (int): image height
            bands (list): list of bands to use
            scale (int): stabilization scale factor
            mean (array): mean
            std (std): std
            mode (str, optional): train | val | basic. Defaults to 'basic'.
                                    - basic: resize + scale
                                    - val: basic + normalize
                                    - train: val + augmentations    

        Returns:
            list: list of transformations
        """
        transformation_list = [A.Resize(height=height, width=width)]
        
        if mode == "train":
            transformation_list.extend([
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1)
            ])
                
        transformation_list.append(ToTensorV2())
        
        return A.Compose(transformation_list)
    
    def _get_remap_classes_ESAWorldCover(self, classes=set()):
        """Create dictionary from ESAWorldCover to model labels

        Args:
            classes (list, optional): List of ESAWorldCover classes. Defaults to set().

        Returns:
            dict: remap dictionary
        """
        
        exclude = ALL_CLASSES_SET - classes
        
        BASE_LABEL = 0
        label = 0 if len(exclude) == 0 else 1
        
        remap = {}
        for c in ESAWorldCover:
            if c in exclude:
                remap[c] = BASE_LABEL
            else:
                remap[c] = label
                label += 1
        
        return remap
    
    def get_pytorch_datasets(self):
        """Get pytorch DataSets for training and validation

        Returns:
            tuple: train_set, val_set
        """
        if self.train_set is None or self.val_set is None:
             
            t_train = self._get_transformations(self.width, self.height, mode='train')
            t_val   = self._get_transformations(self.width, self.height, mode='val')

            class_remap = self._get_remap_classes_ESAWorldCover(self.classes)
            
            self.train_set = SentinelDataset(self.X_train, self.y_train, bands=self.bands, 
                                        transforms=t_train, label_mappings=class_remap,
                                        scale=self.scale, mean=self.mean, std=self.std, 
                                        use_level_C1=self.use_level_C1,
                                        improved_mask=self.improved_mask)
            
            self.val_set = SentinelDataset(self.X_val, self.y_val, bands=self.bands, 
                                        transforms=t_val, label_mappings=class_remap,
                                        scale=self.scale, mean=self.mean, std=self.std,  
                                        use_level_C1=self.use_level_C1,
                                        improved_mask=self.improved_mask)
        
        return self.train_set, self.val_set

    def create_rgb_image(self, image):
        """Create RGB image from normalized image from dataloader

        Args:
            image (array): Normalized image from dataloader

        Returns:
            array: RGB image
        """ 
        
        if self.use_level_C1:
            image = np.delete(image, 10, 0)
            
        image = image.transpose(1,2,0)
        
        mask = np.sign(image)
        image = image * self.std[self.bands] + self.mean[self.bands]
        image = image * mask
        image /= np.max(image)

        if [Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2] == self.bands:
            return image
        else:
            return image[:, :, [Sentinel2Bands.B4, Sentinel2Bands.B3, Sentinel2Bands.B2]]
    
    def create_mask_difference_image(self, pred_mask, target_mask, input_rgb_image):
        """Create RGB image of difference between target and prediction masks

        Args:
            pred_mask (array): prediction mask
            target_mask (array): target mask
            input_rgb_image (array): Input RGB image - used only to identify non-zero pixels

        Returns:
            array: rgb image
        """
        # change order of pytorch tensor dimensions
        mask = input_rgb_image == 0

        res = np.zeros((target_mask.shape[0], target_mask.shape[1],3))
        # Green for true positive
        res[target_mask == pred_mask] = [1, 1, 1]
        # Red for false positive, false negative, true negative
        res[target_mask != pred_mask] = [1, 0, 0] 

        res[mask] = 0
        
        return res 
        
    def create_distance_masks(self, mask_list, output_path):
        
        if not os.path.exists(f'{output_path}/distance'):
            os.mkdir(f'{output_path}/distance')
        
        weight_masks_list = []  
        for mask_path in mask_list:
            mask = tifffile.imread(mask_path)
            dist = self._compute_distance_mask(mask, self.classes)
            
            mask_name = os.path.split()[-1][:-len('.tif')]
            name = f'DIST_{mask_name}_{self.hash}.npy'
            dist_path = f'{output_path}/distance/{name}.npy'
            weight_masks_list.append(dist_path)
            
            np.save(dist_path, dist)
        
        return weight_masks_list
        
    def _image_zero_ratio(self, image_path):
        
        image = tifffile.imread(image_path)
        image = image.transpose(1,2,0)
        image = image.sum(axis=2)
        no_information = np.logical_or(image == 0, np.isnan(image))

        zero_pixels = np.sum(no_information)

        return zero_pixels / (self.width * self.height) 
        
    
    def _compute_distance_mask(self, mask, classes):
        
        result = np.zeros(mask.shape)
        
        for c in classes:
            input_mask = np.zeros(mask.shape) 
            input_mask[mask == c] = 1
            
            dist = distance_transform_edt(input_mask)
            
            result += dist
        
        return result