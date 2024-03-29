## Inference with pretrained models

We provide testing scripts to evaluate a KITTI dataset.

### Test a dataset

- single GPU
- CPU
- single node multiple GPU
- multiple node

You can use the following commands to test a dataset.

```shell
# single-gpu testing (you may need to set PYTHONPATH)
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# CPU: disable GPUs and run single-gpu testing script (you may need to set PYTHONPATH)
export CUDA_VISIBLE_DEVICES=-1
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```
  
Optional arguments:

- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: In this version, we only utilize common metrics in depth estimation, such as Abs Rel, RMSE, *etc*. In the feature, we will consider more challengeable 3D metrics. Hence, this arg is in redundancy, where you can pass any character.
- `--show`: If specified, depth results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment, otherwise you may encounter the error like `cannot connect to X server`.
- `--show-dir`: If specified, depth results will be plotted on the images and saved to the specified directory. It is only applicable to single GPU testing and used for debugging and visualization. You do NOT need a GUI available in your environment for using this option.
- `--eval-options`: Optional parameters for `dataset.format_results` and `dataset.evaluate` during evaluation. Unuseful right now.

Examples:

Assume that you have already downloaded the checkpoints to the directory `checkpoints/`.

1. Test TIE-KD and visualize the results. Press any key for the next image.

    ```shell
    python tools/test.py configs/students/ours_mobile2_adabins.py \
        checkpoints/ours_mobile2_adabins.pth \
        --show
    ```

2. Test TIE-KD and save the painted images for latter visualization.

    ```shell
    python tools/test.py configs/students/ours_mobile2_adabins.py \
        checkpoints/ours_mobile2_adabins.pth \
        --show-dir ours_mobile2_adabins_results
    ```

3. Test TIE-KD on KITTI (without saving the test results) and evaluate the mIoU.

    ```shell
    python tools/test.py configs/students/ours_mobile2_adabins.py \
        checkpoints/ours_mobile2_adabins.pth \
        --eval x(can be any arg)
    ```

4. Test TIE-KD with 4 GPUs, and evaluate the standard 2D metric.

    ```shell
    bash ./tools/dist_test.sh configs/students/ours_mobile2_adabins.py \
        checkpoints/ours_mobile2_adabins.pth \
        4 --eval x(can be any arg)
    ```

