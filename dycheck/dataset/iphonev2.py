import cv2
import os.path as osp
from pathlib import Path
import numpy as np
import json
import copy
from torch.utils.data import Dataset
from typing import Optional, Union
from PIL import Image
from . import util

def load_json(filename, **kwargs):
    with open(filename) as f:
        return json.load(f, **kwargs)

def load_img(
    filename, *, use_rgb: bool = True, **kwargs
) -> np.ndarray:
    img = cv2.imread(filename, **kwargs)
    if use_rgb and img.shape[-1] >= 3:
        # Take care of RGBA case when flipping.
        img = np.concatenate([img[..., 2::-1], img[..., 3:]], axis=-1)
    return img

class iPhoneDatasetV2(Dataset):
    def __init__(self, data_root, scene, _factor=2):
        super().__init__()
        self.data_root = data_root
        self.scene = scene
        self._load_scene_cameras()
        self.aligned_depth_path = Path(self.data_root) / 'depths' / scene / 'aligned_depth' / f"{_factor}x"
        self.rgb_path = Path(self.data_root) / scene / 'rgb' / f"{_factor}x"

    def _load_scene_cameras(self):
        self.cam0_infos = util.get_colmap_camera_params(sparse_dir= osp.join(self.data_root, self.scene, "flow3/colmap/sparse"))

    def load_aligned_depth(self, camera_id, time_id):
        assert camera_id == 0
        file_name = f"{camera_id}_{str(time_id).zfill(5)}.npy"
        return np.load(self.aligned_depth_path / file_name)

    def load_rgb(self, camera_id, time_id):
        file_name = f"{camera_id}_{str(time_id).zfill(5)}.png"
        image = Image.open(self.rgb_path / file_name)
        return np.array(image.convert('RGB')).astype(np.uint8)

    def warp_a2b(self, time_a_id, camera_a_id, time_b_id, camera_b_id):
        image_a = self.load_rgb(camera_a_id, time_a_id)
        image_b = self.load_rgb(camera_b_id, time_b_id)
        camera_a = self.cam0_infos[camera_a_id][f"{camera_a_id}_{str(time_a_id).zfill(5)}.png"]
        depth_a = self.load_aligned_depth(camera_a_id, time_a_id)
        camera_b = self.cam0_infos[camera_b_id][f"{camera_b_id}_{str(time_b_id).zfill(5)}.png"]

        warped_image, valid_mask = camera_a.project(image_a, depth_a, camera_b)

        return image_a, image_b, warped_image, valid_mask