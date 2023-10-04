# Forest segmentation

## Introduction

TO BE FILLED

## Semantic segmentation

Semantic segmentation is a computer vision task that aims to classify each pixel of an image into a set of predefined classes. In this project, we are interested in the semantic segmentation of forest images. 

The detailed state-of-the-art report can be found [here](./research/papers.md).

## Datasets
Download Sentinel 2 images and ESA WorldCover labels using the Google Earth Engine API

### Data description

**Google Earth Engine** is a cloud-based platform for planetary-scale environmental data analysis. Is is free for research and education purposes. 
To use the Google Earth Engine API, you need to create an account [here](https://signup.earthengine.google.com/#!/).

| Name | Units | Min | Max | Scale | Pixel Size | Wavelength | Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | | | | 0.0001 | 60 meters | 443.9nm (S2A) / 442.3nm (S2B) | Aerosols |
| B2 | | | | 0.0001 | 10 meters | 496.6nm (S2A) / 492.1nm (S2B) | Blue |
| B3 | | | | 0.0001 | 10 meters | 560nm (S2A) / 559nm (S2B) | Green |
| B4 | | | | 0.0001 | 10 meters | 664.5nm (S2A) / 665nm (S2B) | Red |
| B5 | | | | 0.0001 | 20 meters | 703.9nm (S2A) / 703.8nm (S2B) | Red Edge 1 |
| B6 | | | | 0.0001 | 20 meters | 740.2nm (S2A) / 739.1nm (S2B) | Red Edge 2 |
| B7 | | | | 0.0001 | 20 meters | 782.5nm (S2A) / 779.7nm (S2B) | Red Edge 3 |
| B8 | | | | 0.0001 | 10 meters | 835.1nm (S2A) / 833nm (S2B) | NIR |
| B8A | | | | 0.0001 | 20 meters | 864.8nm (S2A) / 864nm (S2B) | Red Edge 4 |
| B9 | cm | | | 0.0001 | 60 meters | 945nm (S2A) / 943.2nm (S2B) | Water vapor |
| B11 | | | | 0.0001 | 20 meters | 1613.7nm (S2A) / 1610.4nm (S2B) | SWIR 1 |
| B12 | | | | 0.0001 | 20 meters | 2202.4nm (S2A) / 2185.7nm (S2B) | SWIR 2 |
| AOT | | | | 0.001 | 10 meters | | Aerosol Optical Thickness |
| WVP | cm | | | 0.001 | 10 meters | | Water Vapor Pressure. The height the water would occupy if the vapor were condensed into liquid and spread evenly across the column. |
| SCL | | 1 | 11 | | 20 meters | | Scene Classification Map (The "No Data" value of 0 is masked out) |
| TCI_R | | | | | 10 meters | | True Color Image, Red channel |
| TCI_G | | | | | 10 meters | | True Color Image, Green channel |
| TCI_B | | | | | 10 meters | | True Color Image, Blue channel |
| MSK_CLDPRB | | 0 | 100 | | 20 meters | | Cloud Probability Map (missing in some products) |
| MSK_SNWPRB | | 0 | 100 | | 10 meters | | Snow Probability Map (missing in some products) |
| QA10 | | | | | 10 meters | | Always empty |
| QA20 | | | | | 20 meters | | Always empty |
| QA60 | | | | | 60 meters | | Cloud mask |




**ESA WorldCover 10m v200** is a global land cover map for 2021 at 10m resolution based on Sentinel-1 and Sentinel-2 data ([link](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200)). Dataset was created for the time span 2021-01-01T00:00:00Z–2022-01-01T00:00:00. It comes with 11 classes:


| Value | Color | Description |
| --- | --- | --- |
| 10 | <span style="background-color:#006400">#006400</span> | Tree cover |
| 20 | <span style="background-color:#ffbb22">#ffbb22</span> | Shrubland |
| 30 | <span style="background-color:#ffff4c">#ffff4c</span> | Grassland |
| 40 | <span style="background-color:#f096ff">#f096ff</span> | Cropland |
| 50 | <span style="background-color:#fa0000">#fa0000</span> | Built-up |
| 60 | <span style="background-color:#b4b4b4">#b4b4b4</span> | Bare / sparse vegetation |
| 70 | <span style="background-color:#f0f0f0">#f0f0f0</span> | Snow and ice |
| 80 | <span style="background-color:#0064c8">#0064c8</span> | Permanent water bodies |
| 90 | <span style="background-color:#0096a0">#0096a0</span> | Herbaceous wetland |
| 95 | <span style="background-color:#00cf75">#00cf75</span> | Mangroves |
| 100 | <span style="background-color:#fae6a0">#fae6a0</span> | Moss and lichen |




### Download script

You can download your data using script `download_sentinel_data.py` with your own parameters. Script arguments are:

```
(env) $ python download_sentinel_data.py -h

Download Sentinel-2 + ESA WorldCover 10m v200 data

positional arguments:
  country               Country name
  start-date            Start date
  end-date              End date

options:
  -h, --help            show this help message and exit
  -c CLOUD_COVER, --cloud-cover CLOUD_COVER
                        Maximum cloud cover percentage
  -s SCALE, --scale SCALE
                        Scale or resolution [m/px]
  -p PATCH_SIZE, --patch-size PATCH_SIZE
                        Patch size
  -o OUTPUT_FOLDER, --output-folder OUTPUT_FOLDER
                        Output folder
  --desc DESC           Description
  --bands BANDS         Sentinel bands separated by comma ","
  --offset OFFSET       Number of patches to skip
  --wihtout-mask        Do not download ESA WorldCover masks
  --without-images      Do not download Sentinel-2 images
```

<!-- or use the default parameters in the script:

```python
NAME="year_2021"
COUNTRY = "San Marino"                  # Country name
START_DATE = '2021-01-01'               # Start date of the time interval
END_DATE = '2021-12-31'                 # End date of the time interval
CLOUD_COVER = 10                        # Maximum cloud cover percentage 
SCALE = 10                              # 10m per pixel 
PATCH_SIZE = 256                        # Size of the image patch im pixels
OUTPUT_FOLDER = f'./data'               # Output folder
SENTINEL_BANDS = ['B2', 'B3', 'B4']     # Sentinel bands
``` -->

Description of the Sentinel 2 bands can be found [here](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#bands).

Downloaded data are stored in the following structure

```
<output_path>
    ├── <country_name name>
    │    ├── <images>
    │    │    ├── <country_name>_image_tile_<#>.tif
    │    │    ├── ...
    │    ├── <masks>
    │    │    ├── <country_name>_mask_tile_<#>.tif  
    │    │    ├── ...
    │    ├── grid_features.json // GeoJSON file with grid features
```


## Docker 

[link](https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/)

Docker script

```
docker run --name pytorch-container --gpus all -it --rm -v $(pwd):/app -v /media/daniel/data:/data --shm-size=12gb  pytorch-gpu
```

## Pretrained models on SSL4EO-S12

SSL4EO-S12 [github](https://github.com/zhu-xlab/SSL4EO-S12) contains several pretrained backbones using [MoCo](./research/papers.md#momentum-contrast-for-unsupervised-visual-representation-learning-linkcitation)


Downloaded two backbones traine with Moco using Sentine 2 satellite L1C data.
L1C data contrain 13 bands, which is different to the Level-2A data used in this work. Band **B10** is missing in 2A data.

### Add band **B10** to the downloaded 2A data

```python
config = Config(
      ...
      band_13 = True,
)
```

when the `band_13` flag  is set, `data.dataset.SentinelDataset` will add band **B10** as zeros to the data.

### Register the backbones

Run function `register_SSL4EO_S12_encoders()` from `utils.py` to register the backbones into `segmentation_models_pytorch` module. For proper use set correct path to the downloaded models in `./resources/backbones`

### Train parameters

Choose one of the following backbones and update the configuration.

| Backbone |  smp registered name  | checkpoint |
| --- | --- | --- |
| ResNet18 | resnet18_S2 | ./resources/backbones/resnet18.pth |
| ResNet50 | resnet50_S2 | ./resources/backbones/resnet50.pth |

```python
config = Config(
      ...
      net_config=[
            'encoder_name': <encoder>
            'encoder_weights': 'SSL4EO-S12'
      ]
)
```




## [Experiments](./src/experiments/experimets.md)