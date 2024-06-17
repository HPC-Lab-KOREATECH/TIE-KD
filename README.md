# TIE-KD
## Introduction
we introduce a novel Teacher-Independent Explainable Knowledge Distillation (TIE-KD) framework that streamlines the knowledge transfer from complex teacher models to compact student networks, eliminating the need for architectural similarity

## Related paper
[TIE-KD: Teacher-Independent and Explainable Knowledge Distillation for Monocular Depth Estimation](https://arxiv.org/abs/2402.14340)

## Installation

Please refer to [get_started.md](docs/get_started.md#installation) [(html)](docs/get_started.html) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) [(html)](docs/dataset_prepare.html) for dataset preparation.

## Get Started

We provide [train.md](docs/train.md) [(html)](docs/train.html) and [inference.md](docs/inference.md) [(html)](docs/inference.html) for the usage of this toolbox. 

<!-- In the future, there will be tutorials for [customizing dataset (TODO)](docs/tutorials/customize_datasets.md), [designing data pipeline (TODO)](docs/tutorials/data_pipeline.md), [customizing modules (TODO)](docs/tutorials/customize_models.md), and [customizing runtime (TODO)](docs/tutorials/customize_runtime.md). We also provide [training tricks (TODO)](docs/tutorials/training_tricks.md). -->

## Results and models

### Teachers

| Model | Backbone | Train Epoch | Abs Rel | RMSE | Config | Download |
| :------: | :--------: | :----: | :--------------: | :------: | :------: | :--------: |
| Adabins  |  EfficientNetB5-AP  |  24   | 0.0593 | 2.3309 |  [config](configs/teachers/adabins_efnetb5ap_kitti_24e.py) |  [model]([https://drive.google.com/file/d/17srI3mFoYLdnN1As4a2fRGrHA0UHuujX/view?usp=sharing](https://koreatechackr-my.sharepoint.com/:u:/g/personal/bluekds_koreatech_ac_kr/Ecb81qU-z39AvaUeN-Up1kcBrWYcowajq83eZAlobCcxNg?e=klJ65F))
| BTS  |  ResNet-50  |  24   | 0.0586 | 2.4798 |  [config](configs/teachers/bts_r50_kitti_24e_b4.py) | [model]([https://drive.google.com/file/d/1VBSYwoyquYAR3sP6sg0qyhaZhC0BKj2v/view?usp=sharing](https://koreatechackr-my.sharepoint.com/:u:/g/personal/bluekds_koreatech_ac_kr/EXkshDfzK1BPqpLSIX4LgtoBusnW1t4HfyP6yUZsPsw4fQ?e=lftblO))
| Depthformer | SwinL-w7-22k   |  24   | 0.0513 | 2.1038 |  [config](configs/teachers/depthformer_swinl_22k_w7_kitti.py) | [model]([https://drive.google.com/file/d/1wFcF8G8x3WwDj9owJ5G47NrUIg--BTKJ/view?usp=sharing](https://koreatechackr-my.sharepoint.com/:u:/g/personal/bluekds_koreatech_ac_kr/EQ2uVeyE5nBMp0HxQ8qhBMwBGUdXnjK4bdU2P7cBb6VJBA?e=r1CWgY))

### Students

| Teacher | Method | Loss | Backbone | Train Epoch | Abs Rel | RMSE | Config | Download |
| :------: | :------: | ------ | :--------: | :----: | :--------------: | :------: |  :------: | :--------: |
| - | Baseline | SI | MobileNetV2   |  24   | 0.0663 | 2.5625 |  [config](configs/students/baseline_mobile2.py) | [model](https://drive.google.com/file/d/1vrFupMPmwhFcMYOfcDrpZgnwoRPTzfVD/view?usp=sharing)
| - | Baseline | SI | ResNet18   |  24   | 0.0634 | 2.5311 |  [config](configs/students/baseline_resnet18.py) | [model](https://drive.google.com/file/d/1i01BTingD2QnheZabCBkvVO04Nd-wEkY/view?usp=sharing)
| - | Baseline | SI | ResNet50   |  24   | 0.0605 | 2.4159 |  [config](configs/students/baseline_resnet50.py) | [model](https://drive.google.com/file/d/1JejvH_wBhga_cjjYPqlru3JkiWvi2pAO/view?usp=sharing)
| Adabins | Res-KD | SSIM | MobileNetV2   |  24   | 0.0697 | 2.5639 |  [config](configs/students/res-kd_mobile_SSIM_adabins.py) | [model](https://drive.google.com/file/d/1xKYXVLn5aKmnpr550WERGEAnvhWwPoUu/view?usp=sharing)
| Adabins | Res-KD | MSE | MobileNetV2   |  24   | 0.0786 | 2.6964 |  [config](configs/students/res-kd_mobile_mse_adabins.py) | [model](https://drive.google.com/file/d/1nrK4gl7pcGtFLPn_UUnU9yKKLvEbj7Ve/view?usp=sharing)
| Adabins | Res-KD | SI | MobileNetV2   |  24   | 0.0739 | 2.7371 |  [config](configs/students/res-kd_mobile_SI_adabins.py) | [model](https://drive.google.com/file/d/1imQLKJ75pQt5OCYrue5vNmpL5jO5gCvA/view?usp=sharing)
| Adabins | Res-KD | SSIM, SI | MobileNetV2   |  24   | 0.0701 | 2.5833 |  [config](configs/students/res-kd_mobile_SSIM_SI_adabins.py) | [model](https://drive.google.com/file/d/1xbPF8KfrD8JXpuZcFI8sgF0_9HZbahOn/view?usp=sharing)
| Adabins | Res-KD | SSIM, MSE | MobileNetV2   |  24   | 0.0808 | 2.6943 |  [config](configs/students/res-kd_mobile_SSIM_mse_adabins.py) | [model](https://drive.google.com/file/d/10IL5Ak219kEiCs-yczBUj6ujB7S3Uwft/view?usp=sharing)
| Adabins | TIE-KD | L_DPM | MobileNetV2   |  24   | 0.0718 | 2.5433 |  [config](configs/students/ours_mobile_kl_only_adabins.py) | [model](https://drive.google.com/file/d/16b6Vv08U75fY5T7frZ_54161i6_2QAlL/view?usp=sharing)
| Adabins | TIE-KD | L_DEPTH | MobileNetV2   |  24   | 0.0696 | 2.4646 |  [config](configs/students/ours_mobile_SSIM_only_adabins.py) | [model](https://drive.google.com/file/d/1GLwdhgm-cIF0KBdFEINVoRd8EMun7kY5/view?usp=sharing)
| Adabins | TIE-KD | L_DPM, L_DEPTH | MobileNetV2   |  24   | 0.0654 | 2.4315 |  [config](configs/students/ours_mobile2_adabins.py) | [model](https://drive.google.com/file/d/1wdcVqNNFtyI3vviioQABH-v2NZeLSoCM/view?usp=sharing)
| BTS | Res-KD | SSIM | MobileNetV2   |  24   | 0.0697 | 2.6357 |  [config](configs/students/res-kd_mobile_SSIM_bts.py) | [model](https://drive.google.com/file/d/1rbTppeku1q_AQRyNVdzxV1L2rDDFbrAl/view?usp=sharing)
| BTS | Res-KD | MSE | MobileNetV2   |  24   | 0.0820 | 2.7440 |  [config](configs/students/res-kd_mobile_mse_bts.py) | [model](https://drive.google.com/file/d/1Ax8VlR0durSeYMiSg5Wy_Ce8aymr9MsW/view?usp=sharing)
| BTS | Res-KD | SI | MobileNetV2   |  24   | 0.0782 | 2.8106 |  [config](configs/students/res-kd_mobile_SI_bts.py) | [model](https://drive.google.com/file/d/1BsMaoJiUu5TXlur9sJp09OVo20kpvUYX/view?usp=sharing)
| BTS | Res-KD | SSIM, SI | MobileNetV2   |  24   | 0.0690 | 2.6168 |  [config](configs/students/res-kd_mobile_SSIM_SI_bts.py) | [model](https://drive.google.com/file/d/17CmKkXgrrDul2OIF70Py6svIpNLm354f/view?usp=sharing)
| BTS | Res-KD | SSIM, MSE | MobileNetV2   |  24   | 0.0914 | 2.7983 |  [config](configs/students/res-kd_mobile_SSIM_mse_bts.py) | [model](https://drive.google.com/file/d/1HdtqNRCxXDgpAQO5vt6m7sQ_c9henLOk/view?usp=sharing)
| BTS | TIE-KD | L_DPM | MobileNetV2   |  24   | 0.0722 | 2.6459 |  [config](configs/students/ours_mobile_kl_only_bts.py) | [model](https://drive.google.com/file/d/1T-ltkIKTtcsINJdSBpXE9k1PbaElWFQ5/view?usp=sharing)
| BTS | TIE-KD | L_DEPTH | MobileNetV2   |  24   | 0.0679 | 2.5694 |  [config](configs/students/ours_mobile_SSIM_only_bts.py) | [model](https://drive.google.com/file/d/1TTiqdK5GcTBOvBmCdOwid50VfmcRFWi2/view?usp=sharing)
| BTS | TIE-KD | L_DPM, L_DEPTH | MobileNetV2   |  24   | 0.0656 | 2.4984 |  [config](configs/students/ours_mobile2_bts.py) | [model](https://drive.google.com/file/d/1Kd0chJIpPCdDLK_3UX1thNU24ggPL3B2/view?usp=sharing)
| Depthformer | Res-KD | SSIM | MobileNetV2   |  24   | 0.0692 | 2.5009 |  [config](configs/students/res-kd_mobile_SSIM_depthformer.py) | [model](https://drive.google.com/file/d/10pOSw6QH3047_SRzwgBVundoLq7Y_f69/view?usp=sharing)
| Depthformer | Res-KD | MSE | MobileNetV2   |  24   | 0.0805 | 2.6029 |  [config](configs/students/res-kd_mobile_mse_depthformer.py) | [model](https://drive.google.com/file/d/1_1lxxFeXSVKovIDOjmWU3vs_TFdQzfsZ/view?usp=sharing)
| Depthformer | Res-KD | SI | MobileNetV2   |  24   | 0.0724 | 2.6717 |  [config](configs/students/res-kd_mobile_SI_depthformer.py) | [model](https://drive.google.com/file/d/1PNm4djPKZGRXERDx01aU2UzTikez_lcZ/view?usp=sharing)
| Depthformer | Res-KD | SSIM, SI | MobileNetV2   |  24   | 0.0682 | 2.5709 |  [config](configs/students/res-kd_mobile_SSIM_SI_depthformer.py) | [model](https://drive.google.com/file/d/1HV8Wf8cpeSIN_HW9iSVcx92sMmSsYyIN/view?usp=sharing)
| Depthformer | Res-KD | SSIM, MSE | MobileNetV2   |  24   | 0.0770 | 2.6391 |  [config](configs/students/res-kd_mobile_SSIM_mse_depthformer.py) | [model](https://drive.google.com/file/d/1NBW_YZUBRk3SzoLJe6K_5N2pxa2nEqHN/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM | MobileNetV2   |  24   | 0.0713 | 2.5241 |  [config](configs/students/ours_mobile_kl_only_depthformer.py) | [model](https://drive.google.com/file/d/1CbKjtETaI_eoYeGyZ8y1_b8QsRg1DO0i/view?usp=sharing)
| Depthformer | TIE-KD | L_DEPTH | MobileNetV2   |  24   | 0.0698 | 2.4805 |  [config](configs/students/ours_mobile_SSIM_only_depthformer.py) | [model](https://drive.google.com/file/d/1fAqLjzOCwpOErshyRyUzF-IEamPJKGEe/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM, L_DEPTH | MobileNetV2   |  24   | 0.0657 | 2.4402 |  [config](configs/students/ours_mobile2_depthformer.py) | [model](https://drive.google.com/file/d/1V_oaOn1Gbs76_nR1hKMwMpYX0NqI-Jzu/view?usp=sharing)

| Teacher | Method | Loss | Backbone | Train Epoch | Abs Rel | RMSE | Config | Download |
| :------: | :------: | ------ | :--------: | :----: | :--------------: | :------: |  :------: | :--------: |
| Adabins | TIE-KD | L_DPM, L_DEPTH | ResNet18   |  24   | 0.0628 | 2.4029 |  [config](configs/students/ours_r18_adabins.py) | [model](https://drive.google.com/file/d/1Y6szcbqMRI6CQbJ94jUWuYBMBTUrjPC5/view?usp=sharing)
| Adabins | TIE-KD | L_DPM, L_DEPTH | ResNet50   |  24   | 0.0597 | 2.3060 |  [config](configs/students/ours_r50_adabins.py) | [model](https://drive.google.com/file/d/1YOCyEJ-9UmSVbUbaTHcoNFL-CrEIy2Xg/view?usp=sharing)
| BTS | TIE-KD | L_DPM, L_DEPTH | ResNet18   |  24   | 0.0635 | 2.4527 |  [config](configs/students/ours_r18_bts.py) | [model](https://drive.google.com/file/d/1YGmK6TuU_n4Lr2cGGgFxX7LEY9UkY6Oz/view?usp=sharing)
| BTS | TIE-KD | L_DPM, L_DEPTH | ResNet50   |  24   | 0.0615 | 2.4019 |  [config](configs/students/ours_r50_bts.py) | [model](https://drive.google.com/file/d/1OIeUE2oijsDy_-ogMazgcxf8lX41k25M/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM, L_DEPTH | ResNet18   |  24   | 0.0624 | 2.3963 |  [config](configs/students/ours_r18_depthformer.py) | [model](https://drive.google.com/file/d/10cARevwL0h3Way9FdJTr7WkPzvopdGVX/view?usp=sharing)
| Depthformer | TIE-KD | L_DPM, L_DEPTH | ResNet50   |  24   | 0.0586 | 2.2821 |  [config](configs/students/ours_r50_depthformer.py) | [model](https://drive.google.com/file/d/1ENzaTxa1ieOy5t6-MMDmVaYbGGEgzApY/view?usp=sharing)

## Viewer
We provide a viewer that can simultaneously check the results of different methods for three models.
- Move between images with '←' and '→' keys
- Display diff of other images for that image with 1, 2, 3, 4, 5, 6, 7, 8, 9 numeric keys (press the same button again to toggle back)

|[AdaBins](https://hpc-lab-koreatech.github.io/TIE-KD?targets=Teacher_adabins*ver919_kitti_adabins_cal_range_w0.1_beta_4*ver925_base_up7*ver650_KD_SSIM*ver652_res-kd_mobile_mse_adabins*ver651_res-kd_mobile_SI_adabins*ver659_KD_SSIM_SIG*ver912_res-kd_mobile_SSIM_mse_adabins)|
|[BTS](https://hpc-lab-koreatech.github.io/TIE-KD?targets=Teacher_bts*ver926_ours_bts_*ver925_base_up7*ver650_KD_SSIM_bts*ver652_res-kd_mobile_mse_bts*ver651_res-kd_mobile_SI_bts*ver659_KD_SSIM_SIG_bts*ver912_res-kd_mobile_SSIM_mse_bts)|
|[Depthformer](https://hpc-lab-koreatech.github.io/TIE-KD?targets=Teacher_depthformer*ver919_kitti_depthformer_cal_range_w0.1_beta_4*ver925_base_up7*ver650_KD_SSIM_depthformer*ver652_res-kd_mobile_mse_depthformer*ver651_res-kd_mobile_SI_depthformer*ver659_KD_SSIM_SIG_depthformer*ver912_res-kd_mobile_SSIM_mse_depthformer)|

## Acknowledgement
This repo benefits from [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/). Please also consider citing them.

### Special thanks
[@refracta](https://github.com/refracta) - Developing a viewer to check the results, Qualitative and quantitative comparison of data for data selection.
