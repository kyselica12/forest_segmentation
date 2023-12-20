import glob
import os
import ee
import geemap
import json
import argparse
import logging
import re
import json

parser = argparse.ArgumentParser(description='Download Sentinel-2 for Satlas super-resolution')
parser.add_argument('country', type=str, help='Country for which the data will be downloaded unless the grid is provided.')
parser.add_argument('start_date', type=str, help='Start date in format YYYY-MM-DD')
parser.add_argument('end_date', type=str, help='End date in format YYYY-MM-DD')
parser.add_argument('-N',type=int, help='Number of images to download per tile', default=6)
parser.add_argument('--grid', type=str, help='Grid json file. This will be prefered to the country.', default=None)
parser.add_argument('-p', '--patch-size', type=int, help='Patch size in px', default=256)
parser.add_argument('-o', '--output-folder', type=str, help='Output folder', default='.')
parser.add_argument('--desc', type=str, help='Additional describtion', default='')
parser.add_argument('--offset', type=int, help='Number of patches to skip', default=0)
parser.add_argument('--end', type=int, help='Index of the last patch to process. For the purpose of parallel execution.', default=None)

args = parser.parse_args()

NAME = args.desc
COUNTRY = args.country
START_DATE = args.start_date #"2018-03-20"  #args.start_date
END_DATE = args.end_date # "2018-06-20" #args.end_date
GRID_JSON = args.grid #"wallonia_grid.json" 
PATCH_SIZE = args.patch_size
OUTPUT_FOLDER = args.output_folder
OFFSET = args.offset
IDX = OFFSET
END_IDX = args.end
N_IMAGES_PER_TILE = args.N  

BANDS = ['B4','B3','B2']
CRS = "EPSG:3857"
SCALE = 10 # 10m per pixel
N_DIGITS = 4


logging.basicConfig(
    filename=f'{os.path.split(__file__)[0]}/download.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO
)

logging.info(f"DOWNLOAD DATA FOR SATLAS SUPER-RESOLUTION: {args}")

OUTPUT_FOLDER = f'{OUTPUT_FOLDER}/{NAME}'
data_folder = f"{OUTPUT_FOLDER}/sentinel_images"
os.makedirs(data_folder, exist_ok=True)

print("Initializing Google Earth Engine...")
# ee.Authenticate() # google earth engine authentication - project needs to be set up
ee.Initialize()

if GRID_JSON is None:
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    roi = countries.filter(ee.Filter.eq('country_na', COUNTRY))
    grid = roi.geometry().coveringGrid("EPSG:3857", SCALE*PATCH_SIZE) # the 
else:
    with open(GRID_JSON, "r") as f_json:
        grid_json = json.load(f_json)
    grid = ee.FeatureCollection(grid_json)

if not os.path.exists(f'{OUTPUT_FOLDER}/grid.json'):
    with open(f'{OUTPUT_FOLDER}/grid.json', 'w') as f:
        json.dump(grid.getInfo(), f)

data_grid = grid.toList(1, 110)
tile_collection = ee.FeatureCollection(data_grid)

def add_coverage_metadata(image, aoi, scale):
    tmp = image.select('B2')
    toPixels = ee.Number(tmp.unmask(1).reduceRegion(
        reducer=ee.Reducer.count(),
        scale = scale,
        geometry = aoi,
        crs = CRS,
    ).values().get(0))
    cover = toPixels.divide(ee.Number(PATCH_SIZE**2)).multiply(100)
    return image.set({"COVER_PERCENTAGE": cover})

def add_quality_metadata(image):
    clouds = ee.Image(image.get('HIGH_PROBA_CLOUDS_PERCENTAGE'))
    area = ee.Number(image.get('COVER_PERCENTAGE'))

    non_clouds = ee.Number(100).subtract(clouds)
    
    index = area.multiply(non_clouds).divide(100) # percentage of non-clouds pixels in the image

    return image.set({"QUALITY_INDEX": index})

    
image_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
        .filterDate(START_DATE, END_DATE)\
        .filterBounds(grid.geometry()) \
        .select(BANDS)

def download_one_tile(grid_tile):
    tile_collection = ee.FeatureCollection(grid_tile)

    images_col = image_collection.filterBounds(tile_collection.geometry()) \
        .map(lambda img: add_coverage_metadata(img, tile_collection.geometry(), SCALE)) \
        .map(add_quality_metadata) \
        .sort("QUALITY_INDEX", False)
    
    images = ee.ImageCollection(images_col.toList(N_IMAGES_PER_TILE))

    tile_id = IDX + 1 # start indexing from 1
    if int(images.size().getInfo()) > N_IMAGES_PER_TILE:
        logging.warning(f"Not enouth images for tile {tile_id}: available {int(images.size().getInfo())}, required {N_IMAGES_PER_TILE}.")
        return
    
    geemap.download_ee_image_collection(
        collection=images,
        out_dir=data_folder,
        region=tile_collection.geometry(),
        scale=10,
        crs= CRS,
        filenames=[f"image_{tile_id}_{i}" for i in range(N_IMAGES_PER_TILE)]
    )
    

tmp_grid_json = grid.getInfo()
END_IDX = END_IDX if END_IDX is not None else len(grid.getInfo()['features'])

for tile_feature in grid.getInfo()['features'][IDX:END_IDX]:
    tmp_grid_json['features'] = [tile_feature]
    download_one_tile(tmp_grid_json)
    IDX += 1













