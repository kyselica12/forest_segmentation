
"""
This script generates high resolution maps for a given tile index. It loads data from a geodatabase file and creates a high resolution map for each tile. The high resolution map is saved as a NumPy array in a TIFF file. The output files are saved in a directory specified by the user.

Functions:
    load_from_gdb(path_to_bdb, coords): Loads data from a geodatabase file.
    test_all_tiles(path_to_grid_json, path_to_high_resolution_map): Tests all tiles in a grid.
    get_high_resolution_map_for_tile(idx, path_to_grid_json, path_to_high_resolution_map, resolution): Generates a high resolution map for a given tile index.
"""
import os

import tqdm
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import Polygon
import json
import numpy as np
import tifffile
import rasterio
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube

import numpy as np
import geopandas as gpd
from shapely.geometry import box

def load_from_gdb(path_to_bdb, coords):
    """
    Load data from a geodatabase file located at `path_to_bdb` within the bounding box defined by `coords`.
    Only features with the 'WALOUSMAJT' attribute equal to 'Sylviculture' or 'Sapins de Noël' are returned.
    
    Args:
    - path_to_bdb (str): path to the geodatabase file
    - coords (list of tuples): list of (lat, lon) tuples defining the bounding box
    
    Returns:
    - gdf (GeoDataFrame): a GeoDataFrame containing the selected features
    """
    coords = np.array(coords)     
    
    bbox = box(np.min(coords[:, 0]), np.min(coords[:, 1]),np.max(coords[:, 0]), np.max(coords[:, 1]) )

    tmp = gpd.GeoDataFrame({'geometry': [bbox]}, crs=3857)
    tmp = tmp.to_crs(3812)
    bbox_3857 = tmp.loc[0]['geometry']

    gdf = gpd.read_file(path_to_bdb, bbox=bbox_3857)
    gdf.to_crs(3857, inplace=True)

    gdf = gdf[(gdf['WALOUSMAJT'] == 'Sylviculture') | (gdf['WALOUSMAJT'] == 'Sapins de Noël')]

    return gdf

def test_all_tiles(path_to_grid_json, path_to_high_resolution_map):
    """
    Tests all tiles in a grid to see if they contain data from a high resolution map.

    Args:
        path_to_grid_json (str): The path to the grid JSON file.
        path_to_high_resolution_map (str): The path to the high resolution map.

    Returns:
        list: A list of indices of tiles that contain data from the high resolution map.
    """
    with open(path_to_grid_json) as f:
        grid = json.load(f)
    
    wallonia_indices = []    
    for idx in tqdm.tqdm(range(len(grid['features']))):
        coords = grid['features'][idx]['geometry']['coordinates'][0]
        gdf = load_from_gdb(path_to_high_resolution_map, coords)
        if len(gdf) == 0:
            continue
        wallonia_indices.append(idx)
    
    return wallonia_indices

def get_high_resolution_map_for_tile(idx, path_to_grid_json, path_to_high_resolution_map, resolution):
    """
    Generates a high resolution map for a given tile index.

    Args:
        idx (int): Index of the tile for which to generate the high resolution map.
        path_to_grid_json (str): Path to the grid JSON file.
        path_to_high_resolution_map (str): Path to the high resolution map file.
        resolution (float): Resolution of the high resolution map.

    Returns:
        numpy.ndarray: The high resolution map as a NumPy array.
    """

    with open(path_to_grid_json) as f:
        grid = json.load(f)
    coords = grid['features'][idx]['geometry']['coordinates'][0]


    gdf = load_from_gdb(path_to_high_resolution_map, coords)
    gdf["LABEL"] = 10 # CONSISTENT WITH ESA WORLD COVER LABELS

    out_grid = make_geocube(
        vector_data=gdf,
        measurements=["LABEL"],
        resolution=(-resolution, resolution),
        geom={
            "type": "Polygon", 
            "coordinates": [coords],
            "crs": {
                "properties": {"name": "EPSG:3857"}
            }
        },
        fill=0
    )

    out_grid['LABEL'].rio.to_raster('/tmp/high_resolution_map.tif')
    
    img = tifffile.imread('/tmp/high_resolution_map.tif')

    #delete the temporary file
    os.remove('/tmp/high_resolution_map.tif')
    
    return img

if __name__ == "__main__":
    
    PATH = "/media/daniel/data1/sentinel2/WAL_UTS__2018_FILEGDB_3812/WAL_UTS_2018.gdb"
    JSON_PATH = "../wallonie_grid.json"
    RESOLUTION = 10
    SIZE = 512
    OUTPUT_PATH = "/media/daniel/data1/sentinel2/Wallonia_for_SatlasSR/labels"

    OUTPUT_PATH = f"{OUTPUT_PATH}/resolution_{RESOLUTION}_size_{SIZE}"

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    import glob

    for mask_path in tqdm.tqdm(glob.glob(f"{OUTPUT_PATH}/*.tif")):
        mask = tifffile.imread(mask_path)
        mask[mask == 1] = 10
        tifffile.imwrite(mask_path, mask)
    
    # with open(JSON_PATH) as f:
    #     grid = json.load(f)
    #     N = len(grid['features'])
    
    # for idx in tqdm.tqdm(range(N)):
    #     img = get_high_resolution_map_for_tile(idx, JSON_PATH, PATH, RESOLUTION)

    #     if SIZE > img.shape[0]:
    #         for i, x in enumerate(range(0, img.shape[0], SIZE)):
    #             for j, y in enumerate(range(0, img.shape[1], SIZE)):
    #                 sub_img = img[x:x+SIZE, y:y+SIZE]
    #                 tifffile.imwrite(f"{OUTPUT_PATH}/mask_{idx}_{i}_{j}.tif", sub_img)
    #     else:
    #         tifffile.imwrite(f"{OUTPUT_PATH}/mask_{idx}.tif", img)
                    




