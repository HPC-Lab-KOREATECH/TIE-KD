# TIE-KD
## Introduction
we introduce a novel Teacher-Independent Explainable Knowledge Distillation (TIE-KD) framework that streamlines the knowledge transfer from complex teacher models to compact student networks, eliminating the need for architectural similarity

## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

We provide [train.md](docs/train.md) and [inference.md](docs/inference.md) for the usage of this toolbox. 

<!-- In the future, there will be tutorials for [customizing dataset (TODO)](docs/tutorials/customize_datasets.md), [designing data pipeline (TODO)](docs/tutorials/data_pipeline.md), [customizing modules (TODO)](docs/tutorials/customize_models.md), and [customizing runtime (TODO)](docs/tutorials/customize_runtime.md). We also provide [training tricks (TODO)](docs/tutorials/training_tricks.md). -->

## Results and models

### Teachers

| Model | Backbone | Train Epoch | Abs Rel | RMSE | Config | Download |
| :------: | :--------: | :----: | :--------------: | :------: | :------: | :--------: |
| Adabins  |  EfficientNetB5-AP  |  24   | 0.058 | 2.33 |  [config](adabins_efnetb5ap_kitti_24e.py) | [log](resources/logs/adabins_efnetb5ap_kitti_24e.txt) \| [model](https://drive.google.com/file/d/17srI3mFoYLdnN1As4a2fRGrHA0UHuujX/view?usp=sharing)
| BTS  |  ResNet-50  |  24   | 0.058 | 2.33 |  [config](adabins_efnetb5ap_kitti_24e.py) | [log](resources/logs/adabins_efnetb5ap_kitti_24e.txt) \| [model](https://drive.google.com/file/d/17srI3mFoYLdnN1As4a2fRGrHA0UHuujX/view?usp=sharing)
| Depthformer | SwinL-w7-22k   |  24   | 0.058 | 2.36 |  - | -

### Students

| Teacher | Method | Loss | Backbone | Train Epoch | Abs Rel | RMSE | Config | Download |
| :------: | :------: | ------ | :--------: | :----: | :--------------: | :------: |  :------: | :--------: |
| - | Baseline | SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config] | -
| - | Baseline | SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config] | -
| - | Baseline | SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config] | -
| Adabins | Res-KD | SSIM | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | Res-KD | MSE | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | Res-KD | SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | Res-KD | SSIM, SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | Res-KD | SSIM, MSE | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | TIE-KD | L_DPM | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | TIE-KD | L_DEPTH | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | TIE-KD | L_DPM, L_DEPTH | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | Res-KD | SSIM | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | Res-KD | MSE | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | Res-KD | SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | Res-KD | SSIM, SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | Res-KD | SSIM, MSE | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | TIE-KD | L_DPM | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | TIE-KD | L_DEPTH | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | TIE-KD | L_DPM, L_DEPTH | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | Res-KD | SSIM | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | Res-KD | MSE | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | Res-KD | SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | Res-KD | SSIM, SI | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | Res-KD | SSIM, MSE | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | TIE-KD | L_DEPTH | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM, L_DEPTH | MobileNetV2   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)

| Teacher | Method | Loss | Backbone | Train Epoch | Abs Rel | RMSE | Config | Download |
| :------: | :------: | ------ | :--------: | :----: | :--------------: | :------: |  :------: | :--------: |
| Adabins | TIE-KD | L_DPM, L_DEPTH | ResNet18   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Adabins | TIE-KD | L_DPM, L_DEPTH | ResNet50   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | TIE-KD | L_DPM, L_DEPTH | ResNet18   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| BTS | TIE-KD | L_DPM, L_DEPTH | ResNet50   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM, L_DEPTH | ResNet18   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM, L_DEPTH | ResNet50   |  24   | 0.103 | 0.364 |  [config](adabins_efnetb5ap_nyu_24e.py) | [model](https://drive.google.com/file/d/1NRTWApIrxOjeeN7FdNTTOXV3KOuo_-aC/view?usp=sharing)

## Acknowledgement

This repo benefits from [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/). Please also consider citing them.
