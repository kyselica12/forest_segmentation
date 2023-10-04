import glob
import os
import ee
import geemap
import json
import argparse
import logging
import re

parser = argparse.ArgumentParser(description='Download Sentinel-2 + ESA WorldCover 10m v200 data')
parser.add_argument('country', type=str, help='Country name')
parser.add_argument('start_date', type=str, help='Start date in format YYYY-MM-DD')
parser.add_argument('end_date', type=str, help='End date in format YYYY-MM-DD')
parser.add_argument('-c', '--cloud-cover', type=int, help='Maximum cloud cover percentage', default=10)
parser.add_argument('-s', '--scale', type=int, help='Scale or resolution [m/px]', default=10)
parser.add_argument('-p', '--patch-size', type=int, help='Patch size in px', default=256)
parser.add_argument('-o', '--output-folder', type=str, help='Output folder', default='.')
parser.add_argument('--desc', type=str, help='Additional describtion', default='')
parser.add_argument('--bands', type=str, help='Sentinel bands separated by comma ","', default='B2,B3,B4')
parser.add_argument('--offset', type=int, help='Number of patches to skip', default=0)
parser.add_argument('--without-mask', action='store_false', help='Do not download ESA WorldCover masks')
parser.add_argument('--without-images', action='store_false', help='Do not download Sentinel-2 images')
parser.add_argument('--continue-download', action='store_true', help='Continue downloading from the last patch')

args = parser.parse_args()

NAME = args.desc  #"year_2021"
COUNTRY = args.country #"San Marino"#'Vatican City'#"Luxembourg"#
START_DATE =  args.start_date #'2021-01-01'
END_DATE = args.end_date #'2021-12-31'
CLOUD_COVER = args.cloud_cover #10
SCALE = args.scale #10 # 10m per pixel
PATCH_SIZE = args.patch_size #256
OUTPUT_FOLDER = args.output_folder #f'./data'
SENTINEL_BANDS = args.bands.split(',') #['B2','B3','B4']
OFFSET = args.offset #0
DOWNLOAD_MASKS = args.without_mask
DOWNLOAD_IMAGES = args.without_images
N_DIGITS = 4
CONTINUE_DOWNLOAD = args.continue_download

logging.basicConfig(
    filename=f'{os.path.split(__file__)[0]}/download.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO
)

call =  f"download_sentinel_data.py {args.country} {args.start_date} {args.end_date}"
call += f" -c {args.cloud_cover} -s {args.scale} -p {args.patch_size} -o {args.output_folder} --desc {args.desc} --bands {args.bands} --offset {args.offset}"
call += f"--wihout-mask" if not args.without_mask else ""
call += "--without-images" if not args.without_images else ""
logging.info(call)

# ==================== CREATE OUTPUT FOLDERS ====================
print("Creating output folders...")

data_folder = f"{OUTPUT_FOLDER}/{COUNTRY}_{NAME}"

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

images_folder = f'{data_folder}/images'
if not os.path.exists(images_folder):
    os.makedirs(images_folder)
    
masks_folder = f'{data_folder}/masks'
if not os.path.exists(masks_folder):
    os.makedirs(masks_folder)  

print("Initializing Google Earth Engine...")
# ee.Authenticate() # google earth engine authentication - project needs to be set up
ee.Initialize()

countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
roi = countries.filter(ee.Filter.eq('country_na', COUNTRY))

# create grid patches in the roi
print("Generating and saving grid patches...")
grid = roi.geometry().coveringGrid("EPSG:3857", SCALE*PATCH_SIZE) # the 

N_DIGITS = len(str(grid.size().getInfo()))

with open(f'{data_folder}/grid_features.json', 'w') as f:
    json.dump(grid.getInfo(), f)

    
# ==================== DOWNLOAD SENTINEL DATA ====================

if DOWNLOAD_IMAGES:
    print("Downloading Sentinel-2 data...")

    IMAGE_OFFSET = OFFSET
    IMAGE_OFFSET += 0 if not CONTINUE_DOWNLOAD else len(glob.glob(f'{images_folder}/*.tif'))
    data_grid = grid.toList(grid.size(), IMAGE_OFFSET)
    data_grid = ee.FeatureCollection(data_grid)

    image_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterDate(START_DATE, END_DATE)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',CLOUD_COVER))\
            .filterBounds(roi)
        
    def create_quality_band(image):
        return image.addBands(image.select('MSK_CLDPRB').multiply(-1).rename('quality'))

    # add quality band -> negative cloud probability
    image_collection = image_collection.map(create_quality_band)

    # craete mosaic image with the lowest cloud probability
    image = image_collection.qualityMosaic('quality')
    image = image.select(SENTINEL_BANDS).clip(roi.geometry())

    name = f"{COUNTRY}_image_tile_" + ("" if IMAGE_OFFSET == 0 else f"OFFSET_{IMAGE_OFFSET}_")
    geemap.download_ee_image_tiles(
        image,
        features=data_grid,
        out_dir=images_folder,
        prefix=name,
        scale=SCALE,
        crs="EPSG:3857"
    )
            
# ========================================================================

# ==================== DOWNLOAD ESA WORLD COVER MASKS ====================
if DOWNLOAD_MASKS:
    MASKS_OFFSET = OFFSET
    MASKS_OFFSET += 0 if not CONTINUE_DOWNLOAD else len(glob.glob(f'{masks_folder}/*.tif'))
    mask_grid = grid.toList(grid.size(), MASKS_OFFSET)
    mask_grid = ee.FeatureCollection(mask_grid)

    print("Downloading ESA World Cover masks...")

    land_cover_image = ee.ImageCollection('ESA/WorldCover/v200').first().clip(roi.geometry())

    name = f"{COUNTRY}_mask_tile_" + ("" if MASKS_OFFSET == 0 else f"OFFSET_{MASKS_OFFSET}_")
    geemap.download_ee_image_tiles(
        land_cover_image,
        features=mask_grid,
        out_dir=masks_folder,
        prefix=name,
        scale=SCALE,
        crs="EPSG:3857"
    )

# ========================================================================

# ==================== RENAME FILES ====================
print("Renaming files due to OFFSET ...")

    
for folder, flag in zip([images_folder, masks_folder],[DOWNLOAD_IMAGES, DOWNLOAD_MASKS]):
    if not flag: continue
    
    for path in glob.glob(f'{folder}/*OFFSET*.tif'):
        folder_name, filename = os.path.split(path)
        res= re.findall(r'(\d+)', filename)  
        offset = int(res[-2])
        n = int(res[-1]) + offset
        new_filename = re.split(r'\d+', filename)[0].replace('OFFSET_', '')
        new_filename += "{n:{fill}{width}}.tif".format(n=n, fill='0', width=N_DIGITS)
        new_path = os.path.join(folder_name, new_filename)
        os.rename(path, new_path)


print("Done!")








