# Scripts for downloading Sentinel 2 images using Google Earth Engine

## Prerequisites

1. Install python packages from `requirements.txt`
1. Authenticate your GEE account and project [link](https://developers.google.com/earth-engine/guides/auth)
   The easiest authentication method is using Jupyter notebook

    ```python
    import ee
    ee.Authenticate()
    ```

    This will open a browser window where you can select your GEE account and authenticate it.

## Downloading Sentinel 2 images

Script `download_sentinel.py` downloads Sentinel 2 images using GEE. It downloads images and corresponding ESA WorldCover 10m v200 masks by patches of specified size.

### Usage

```text
Download Sentinel-2 + ESA WorldCover 10m v200 data

positional arguments:
  country               Country name
  start_date            Start date in format YYYY-MM-DD
  end_date              End date in format YYYY-MM-DD

options:
  -h, --help            show this help message and exit
  -c CLOUD_COVER, --cloud-cover CLOUD_COVER
                        Maximum cloud cover percentage
  -s SCALE, --scale SCALE
                        Scale or resolution [m/px]
  -p PATCH_SIZE, --patch-size PATCH_SIZE
                        Patch size in px
  -o OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                        Output folder
  --grid GRID           Json file with grid geometries.
  --desc DESC           Additional describtion
  --bands BANDS         Sentinel bands separated by comma ","
  --offset OFFSET       Number of patches to skip
  --without-mask        Do not download ESA WorldCover masks
  --without-images      Do not download Sentinel-2 images
  --continue-download   Continue downloading from the last patch
```

### Output

Results are saved in the following structure:

```text
output_folder/<desc>
    ├── images/            (Sentinel-2 images)
    |      ├── <country>_image_tile_<tile_id>.tif
    |      └── ...
    ├── masks/             (ESA WorldCover 10m v200 masks) 
    |      ├── <country>_mask_tile_<tile_id>.tif
    |      └── ...
    └── grid_features.json (GeoJSON of tile grid geometries) 
```

Indices start from **1**.

## Downloading Sentinel 2 images for SATLAS-SUPER-RESOLUTION

Script `download_sentinel_series.py` downloads Sentinel 2 images using GEE. For each tile, $N$ images are downloaded with the lowest cloud coverage. The resulting images contain 3 bands in this order: B04 (Red), B03 (Green), and B02 (Blue).

### Usage

```text
Download Sentinel-2 for Satlas super-resolution

positional arguments:
  country               Country for which the data will be downloaded unless the grid is provided.
  start_date            Start date in format YYYY-MM-DD
  end_date              End date in format YYYY-MM-DD

options:
  -h, --help            show this help message and exit
  -N N                  Number of images to download per tile
  --grid GRID           Grid json file. This will be prefered to the country.
  -p PATCH_SIZE, --patch-size PATCH_SIZE
                        Patch size in px
  -o OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                        Output folder
  --desc DESC           Additional describtion
  --offset OFFSET       Number of patches to skip
  --end END             Index of the last patch to process. For the purpose of parallel execution.
```

### Output

Results are saved in the following structure:

```text
output_folder/<desc>
    ├── sentinel_images/  (Sentinel-2 images)
    |      ├── image_<tile_id>_<0-5>.tif
    |      └── ...
    └── grid.json         (GeoJSON of tile grid geometries) 
```

Indexing starts from **1** so the tile numbers corrsponds to the output of `download_sentinel.py`.
