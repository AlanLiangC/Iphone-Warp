import torch
from pathlib import Path
import numpy as np
import sys
sys.path.append('/data/yyang/workspace/magiclidar-dataset')
from dycheck.dataset.iphone import iPhoneDataset
from dycheck.submodules.video_depth_anything_metric.video_depth import VideoDepthAnything as metric_vda
from dycheck.submodules.video_depth_anything.video_depth import VideoDepthAnything as vda

from tqdm import tqdm

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def generate_depth_metric(data_root, scene, vda_encoder='vitl', _factor=2):
    # dataset
    iphone_dataset = iPhoneDataset(data_root=data_root, scene=scene)
    frames = iphone_dataset._load_video(_factor=_factor)

    # model
    model = metric_vda(**model_configs[vda_encoder])
    model.load_state_dict(torch.load(f'dycheck/submodules/video_depth_anything/pretrained_model/metric_video_depth_anything_{vda_encoder}.pth', map_location='cpu'), strict=True)
    model = model.to('cuda').eval()
    depths, _ = model.infer_video_depth(frames, 15, input_size=518, device='cuda')
    # save
    saved_path = Path(data_root) / 'depths' / scene / 'vda_metric_depth' / f"{_factor}x"
    saved_path.mkdir(parents=True, exist_ok=True)
    num_frames = depths.shape[0]
    for i in tqdm(range(num_frames), 'Inferenceing depth with VDA'):
        file_name = f"0_{str(i).zfill(5)}.npy"
        np.save(saved_path / file_name, np.clip(depths[i],0,10))

def generate_depth(data_root, scene, vda_encoder='vitl', _factor=2):
    # dataset
    iphone_dataset = iPhoneDataset(data_root=data_root, scene=scene)
    frames = iphone_dataset._load_video(_factor=_factor)
    T,H,W = frames.shape[:3]
    # model
    model = vda(**model_configs[vda_encoder])
    model.load_state_dict(torch.load(f'dycheck/submodules/video_depth_anything/pretrained_weights/video_depth_anything_{vda_encoder}.pth', map_location='cpu'), strict=True)
    model = model.to('cuda').eval()

    # chunk_num = 50
    # frams_num = frames.shape[0]
    # depths = np.zeros([T,H,W])
    # for i in range(frams_num // chunk_num):
    #     sub_frames = frames[(chunk_num*i):(chunk_num*(i+1))]
    #     sub_depths, _ = model.infer_video_depth(sub_frames, 15, input_size=518, device='cuda')
    #     depths[(chunk_num*i):(chunk_num*(i+1))] = sub_depths
    # if frams_num - chunk_num*(frams_num // chunk_num) > 0:
    #     last_frames = frames[(chunk_num*(frams_num // chunk_num)):]
    #     sub_depths, _ = model.infer_video_depth(last_frames, 15, input_size=518, device='cuda')
    #     depths[(chunk_num*(frams_num // chunk_num)):] = sub_depths

    depths, _ = model.infer_video_depth(frames, 15, input_size=518, device='cuda')
    # # save
    saved_path = Path(data_root) / 'depths' / scene / 'vda_depth' / f"{_factor}x"
    saved_path.mkdir(parents=True, exist_ok=True)
    num_frames = depths.shape[0]
    for i in tqdm(range(num_frames), 'Inferenceing depth with VDA'):
        file_name = f"0_{str(i).zfill(5)}.npy"
        np.save(saved_path / file_name, norm_array(depths[i]))

def norm_array(array):
    return (array - array.min()) / (array.max()-array.min())

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    # metric depth
    # generate_depth_metric(
    #     data_root = 'data/iphone_sm',
    #     scene = 'apple'
    # )
    #depth
    generate_depth(
        data_root = 'data/iphone_sm',
        scene = 'apple'
    )