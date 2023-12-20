import json
import os
from re import T
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tifffile
import tqdm
import glob



def cut_images(input_folder, output_folder, size, transpose=False):

    os.makedirs(output_folder, exist_ok=True)
    
    for filename in tqdm.tqdm(glob.glob(f"{input_folder}/*.tif")):
        
        img: np.ndarray = tifffile.imread(filename)
        name = os.path.basename(filename)[:-len(".tif")]

        if transpose:
            img = img.transpose(1,2,0)
            
        for i, x in enumerate(range(0, img.shape[0], size)):
            for j, y in enumerate(range(0, img.shape[1],size)):
                sub_img = img[x:x+size, y:y+size]

                if transpose:
                    sub_img = sub_img.transpose(2,0,1)
                
                tifffile.imwrite(f"{output_folder}/{name}_{i}_{j}.tif", sub_img)

if __name__ == "__main__":
   
   
   
    if True: 
        # rename images to format <tile>_<x>_<y>.tif 
        # tile is the number of the tile in the grid starting from 1

        import re
        # input_folder = "/media/daniel/data1/sentinel2/Wallonia_for_SatlasSR/labels/resolution_2.5_size_2048"
        # output_folder = "/media/daniel/data1/sentinel2/Wallonia_for_SatlasSR/labels/resolution_2.5_size_512"
        # input_folder = "/media/daniel/data3/sentinel2/Wallonie_SatlasSR/spring_2018/sr_images"
        input_folder = "/media/daniel/data3/sentinel2/Wallonie_SatlasSR/spring_2018/sentinel_images"
        
        for file in tqdm.tqdm(glob.glob(f"{input_folder}/*.tif")):

            directory, name = os.path.split(file)
            indices = re.findall(r'(\d+)', name)

            # assert len(indices) == 3
            # tile, x, y = list(map(int, indices))
            # new_file = f"{directory}/{tile}_{x}_{y}.tif"
            assert len(indices) == 2
            
            tile, x = list(map(int, indices))
            new_file = f"{directory}/image_{tile}_{x}.tif"
            # print(file, new_file)
            # break
            os.rename(file, new_file)

    if False:
        import re
        import numpy as np
        import matplotlib.pyplot as plt
        import os



        # DATASET_PATH = "semantic_segmentation/resources/datasets/a9ede607cb8d23d2101beb0fae2d661b"
        DATASET_PATH = "semantic_segmentation/resources/datasets/1b08783e7301e59b662e2de892c57944"

        assert os.path.exists(DATASET_PATH) == True

        def change_path(img_path):
            img_dir, img_name = os.path.split(img_path)

            ints = list(map(int, re.findall(r'\d+', img_name)))
            assert len(ints) == 3
 
            img_name = f'{ints[0]+1}_{ints[1]}_{ints[2]}.tif'
            
            return os.path.join(img_dir, img_name)

        data = []

        with open(f'{DATASET_PATH}/train.csv', 'r') as f:
            lines = f.readlines()
            
            data.append(lines[0])
            for l in lines[1:]:
                img, mask = l.split(',')
                new_mask = change_path(mask)

                data.append(f'{img},{new_mask}')
            
        text = '\n'.join(data)

        with open(f'{DATASET_PATH}/tmp_train.csv', 'w') as f:
            f.write(text)
    
