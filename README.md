# Iphone-Warp

## Metric depth对齐
> 因为 video depth anything 更新了一个 metric 深度估计的模型，所以可以尝试一下做备选。

### Step 1： 深度估计模型估计深度
- 下载权重文件[https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth]到`dycheck/submodules/video_depth_anything_metric/pretrained_model`

- 选定scene后推理深度: `python tools/generate_depth.py`
    > scene和推理的function均可在文件中更改，此时函数为 generate_depth_metric
    ```
    if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    ################ metric depth
    # generate_depth_metric(
    #     data_root = 'data/iphone_sm',
    #     scene = 'apple'
    # )
    ################ depth
    # generate_depth(
    #     data_root = 'data/iphone_sm',
    #     scene = 'apple'
    # )
    ```

### Step 2： 深度对齐
- metric对齐的代码在 `tools/align_depth.py`中
- example
```
    python tools/align_depth.py \
        --img_dir ***/paper-windmill/rgb/2x \
        --out_raw_dir align深度的存储路径 \
        --out_aligned_dir 深度估计模型的输出路径 \
        --sparse_dir ***/paper-windmill/flow3d_preprocessed/colmap/sparse \
    """
```

### Step 3: Warp
> 写了个脚本检查warp的结果: `tools/warp_iphonev3.py`

- 修改warp时使用depth的路径，在 `dycheck/dataset/iphonev3.py`文件的581行，路径改为Step 2 设置的output的路径

- 也可以通过 `warpped_iphone_v3.ipynb` 更方便的调试

## 视差对齐
> 视差对齐与上面的步骤相同，区别在于Step 1深度估计采用视差模型，然后Step 2采用视差对齐

### Step 1： 深度估计模型估计深度
- 下载权重文件[https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true]到`dycheck/submodules/video_depth_anything/pretrained_model`

- 选定scene后推理深度: `python tools/generate_depth.py`
    > scene和推理的function均可在文件中更改，此时函数为 generate_depth
    ```
    if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    ################ metric depth
    # generate_depth_metric(
    #     data_root = 'data/iphone_sm',
    #     scene = 'apple'
    # )
    ################ depth
    # generate_depth(
    #     data_root = 'data/iphone_sm',
    #     scene = 'apple'
    # )
    ```

### Step 2： 深度对齐
- 视差对齐的代码在 `tools/align_depth_disp.py`中
- example
```
    python tools/align_depth_disp.py \
        --img_dir ***/paper-windmill/rgb/2x \
        --out_raw_dir align深度的存储路径 \
        --out_aligned_dir 深度估计模型的输出路径 \
        --sparse_dir ***/paper-windmill/flow3d_preprocessed/colmap/sparse \
    """
```

### Step 3: Warp
> 写了个脚本检查warp的结果: `tools/warp_iphonev3.py`

- 修改warp时使用depth的路径，在 `dycheck/dataset/iphonev3.py`文件的581行，路径改为Step 2 设置的output的路径

- 也可以通过 `warpped_iphone_v3.ipynb` 更方便的调试