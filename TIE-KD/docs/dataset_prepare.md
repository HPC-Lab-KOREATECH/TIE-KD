## Prepare datasets

It is recommended to symlink the dataset root to `$TIE-KD/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```none
TIE-KD
├── depth
├── tools
├── configs
├── splits
├── data
│   ├── kitti
│   │   ├── input
│   │   │   ├── 2011_09_26
│   │   │   ├── 2011_09_28
│   │   │   ├── ...
│   │   │   ├── 2011_10_03
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── 2011_09_26_drive_0002_sync
│   │   │   ├── ...
│   │   │   ├── 2011_10_03_drive_0047_sync
|   |   ├── benchmark_test
│   │   │   ├── 0000000000.png
│   │   │   ├── 0000000001.png
│   │   │   ├── ...
│   │   │   ├── 0000000499.png
|   |   ├── benchmark_cam
│   │   │   ├── 0000000000.txt
│   │   │   ├── 0000000001.txt
│   │   │   ├── ...
│   │   │   ├── 0000000499.txt
│   │   ├── split_file.txt
```

### **KITTI**

Download the offical dataset from this [link](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), including the raw data (about 200G) and fine-grained ground-truth depth maps. 

Then, unzip the files into data/kitti. Remember to organizing the directory structure following instructions (Only need a few cut operations). 

Finally, copy split files (whose names are started with *kitti*) in splits folder into data/kitti. Here, I utilize eigen splits following other supervised methods.

Some methods may use the camera intrinsic parameters (*i.e.,* BTS), you need to download the [benchmark_cam](https://drive.google.com/file/d/1ktSDTUx9dDViBKoAeqTERTay1813xfUK/view?usp=sharing) consisting of camera intrinsic parameters of the benchmark test set.

