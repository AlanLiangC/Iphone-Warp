import cv2
import os.path as osp
import numpy as np
import json
import copy
from torch.utils.data import Dataset
from typing import Optional, Union

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

def matmul(a, b):
    return a @ b


def matv(a, b) :
    return matmul(a, b[..., None])[..., 0]

def points_to_local_points(
    points: np.ndarray,
    extrins: np.ndarray,
) -> np.ndarray:

    return matv(extrins[..., :3, :3], points) + extrins[..., :3, 3]

def project(
    points,
    intrins,
    extrins,
    radial_distortions = None,
    tangential_distortions = None,
    *,
    return_depth: bool = False,
    use_projective_depth: bool = True,
):

    tensors_to_check = [intrins, extrins]
    if radial_distortions is not None:
        tensors_to_check.append(radial_distortions)
    if tangential_distortions is not None:
        tensors_to_check.append(tangential_distortions)
    if isinstance(points, np.ndarray):
        assert all([isinstance(x, np.ndarray) for x in tensors_to_check])
        np_or_jnp = np
    else:
        raise NotImplementedError

    local_points = points_to_local_points(points, extrins)

    normalized_pixels = np_or_jnp.where(
        local_points[..., -1:] != 0,
        local_points[..., :2] / local_points[..., -1:],
        0,
    )
    r2 = (normalized_pixels**2).sum(axis=-1, keepdims=True)

    if radial_distortions is not None:
        # Apply radial distortion.
        radial_scalars = 1 + r2 * (
            radial_distortions[..., 0:1]
            + r2
            * (
                radial_distortions[..., 1:2]
                + r2 * radial_distortions[..., 2:3]
            )
        )
    else:
        radial_scalars = 1

    if tangential_distortions is not None:
        # Apply tangential distortion.
        tangential_deltas = 2 * tangential_distortions * np_or_jnp.prod(
            normalized_pixels,
            axis=-1,
            keepdims=True,
        ) + tangential_distortions[..., ::-1] * (
            r2 + 2 * normalized_pixels**2
        )
    else:
        tangential_deltas = 0

    # Apply distortion.
    normalized_pixels = normalized_pixels * radial_scalars + tangential_deltas

    # Map the distorted ray to the image plane and return the depth.
    pixels = matv(
        intrins,
        np_or_jnp.concatenate(
            [
                normalized_pixels,
                np_or_jnp.ones_like(normalized_pixels[..., :1]),
            ],
            axis=-1,
        ),
    )[..., :2]

    if not return_depth:
        return pixels
    else:
        depths = (
            local_points[..., 2:]
            if use_projective_depth
            else np_or_jnp.linalg.norm(local_points, axis=-1, keepdims=True)
        )
        return pixels, depths

def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
):
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + k3 * r))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + 3.0 * k3 * r)
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y

def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations: int = 10,
):
    """Computes undistorted (x, y) from (xd, yd).

    Note that this function is purely running on CPU and thus could be slow.
    The original Nerfies & HyperNeRF are training on distorted raw images but
    with undistorted rays.
    """
    # Initialize from the distorted point.
    x = xd.copy()
    y = yd.copy()

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(
            np.abs(denominator) > eps,
            x_numerator / denominator,
            np.zeros_like(denominator),
        )
        step_y = np.where(
            np.abs(denominator) > eps,
            y_numerator / denominator,
            np.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return x, y

class Camera:
    def __init__(
        self,
        orientation: np.ndarray,
        position: np.ndarray,
        focal_length: Union[np.ndarray, float],
        principal_point: np.ndarray,
        image_size: np.ndarray,
        skew: Union[np.ndarray, float] = 0.0,
        pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
        radial_distortion: Optional[np.ndarray] = None,
        tangential_distortion: Optional[np.ndarray] = None,
        *,
        use_center: bool = True,
        use_projective_depth: bool = True,
    ):
        """Constructor for camera class."""
        if radial_distortion is None:
            radial_distortion = np.array([0, 0, 0], np.float32)
        if tangential_distortion is None:
            tangential_distortion = np.array([0, 0], np.float32)

        self.orientation = np.array(orientation, np.float32)
        self.position = np.array(position, np.float32)
        self.focal_length = np.array(focal_length, np.float32)
        self.principal_point = np.array(principal_point, np.float32)
        self.image_size = np.array(image_size, np.uint32)

        # Distortion parameters.
        self.skew = np.array(skew, np.float32)
        self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, np.float32)
        self.radial_distortion = np.array(radial_distortion, np.float32)
        self.tangential_distortion = np.array(
            tangential_distortion, np.float32
        )

        self.use_center = use_center
        self.use_projective_depth = use_projective_depth

    @classmethod
    def fromjson(cls, filename):
        camera_dict = load_json(filename)

        # Fix old camera JSON.
        if "tangential" in camera_dict:
            camera_dict["tangential_distortion"] = camera_dict["tangential"]

        return cls(
            orientation=np.asarray(camera_dict["orientation"]),
            position=np.asarray(camera_dict["position"]),
            focal_length=camera_dict["focal_length"],
            principal_point=np.asarray(camera_dict["principal_point"]),
            image_size=np.asarray(camera_dict["image_size"]),
            skew=camera_dict["skew"],
            pixel_aspect_ratio=camera_dict["pixel_aspect_ratio"],
            radial_distortion=np.asarray(camera_dict["radial_distortion"]),
            tangential_distortion=np.asarray(
                camera_dict["tangential_distortion"]
            ),
        )

    @property
    def scale_factor_x(self):
        return self.focal_length

    @property
    def scale_factor_y(self):
        return self.focal_length * self.pixel_aspect_ratio

    @property
    def principal_point_x(self):
        return self.principal_point[0]

    @property
    def principal_point_y(self):
        return self.principal_point[1]

    @property
    def translation(self):
        return -self.orientation @ self.position

    @property
    def optical_axis(self):
        return self.orientation[2, :]

    @property
    def image_shape(self):
        return np.array([self.image_size_y, self.image_size_x], np.uint32)

    @property
    def intrin(self):
        return np.array(
            [
                [self.scale_factor_x, self.skew, self.principal_point_x],
                [0, self.scale_factor_y, self.principal_point_y],
                [0, 0, 1],
            ],
            np.float32,
        )

    @property
    def image_size_y(self):
        return self.image_size[1]

    @property
    def image_size_x(self):
        return self.image_size[0]

    @property
    def distortion(self):

        return np.concatenate(
            [
                self.radial_distortion[:2],
                self.tangential_distortion,
                self.radial_distortion[-1:],
            ]
        )

    @property
    def has_tangential_distortion(self):
        return any(self.tangential_distortion != 0)

    @property
    def has_radial_distortion(self):
        return any(self.radial_distortion != 0)

    @property
    def extrin(self):
        # 4x4 world-to-camera transform.
        return np.concatenate(
            [
                np.concatenate(
                    [self.orientation, self.translation[..., None]], axis=-1
                ),
                np.array([[0, 0, 0, 1]], np.float32),
            ],
            axis=-2,
        )

    def copy(self) -> "Camera":
        return copy.deepcopy(self)

    def rescale_image_domain(self, scale: float) -> "Camera":
        """Rescale the image domain of the camera."""
        if scale <= 0:
            raise ValueError("scale needs to be positive.")

        camera = self.copy()
        camera.focal_length *= scale
        camera.principal_point *= scale
        camera.image_size = np.array(
            (
                int(round(self.image_size[0] * scale)),
                int(round(self.image_size[1] * scale)),
            )
        )
        return camera

    def translate(self, transl: np.ndarray):
        camera = self.copy()
        camera.position += transl
        return camera

    def rescale(self, scale: float):
        if scale <= 0:
            raise ValueError("scale needs to be positive.")

        camera = self.copy()
        camera.position *= scale
        return camera

    def undistort_image_domain(self):

        camera = self.copy()
        camera.skew = 0
        camera.radial_distortion = np.zeros(3, dtype=np.float32)
        camera.tangential_distortion = np.zeros(2, dtype=np.float32)
        return camera

    def project(
        self,
        points: np.ndarray,
        return_depth: bool = False,
        use_projective_depth: Optional[bool] = None,
    ):
        if use_projective_depth is None:
            use_projective_depth = self.use_projective_depth

        return project(
            points,
            self.intrin,
            self.extrin,
            self.radial_distortion,
            self.tangential_distortion,
            return_depth=return_depth,
            use_projective_depth=use_projective_depth,
        )

    def pixels_to_local_viewdirs(self, pixels: np.ndarray):
        y = (pixels[..., 1] - self.principal_point_y) / self.scale_factor_y
        x = (
            pixels[..., 0] - self.principal_point_x - y * self.skew
        ) / self.scale_factor_x

        if self.has_radial_distortion or self.has_tangential_distortion:
            x, y = _radial_and_tangential_undistort(
                x,
                y,
                k1=self.radial_distortion[0],
                k2=self.radial_distortion[1],
                k3=self.radial_distortion[2],
                p1=self.tangential_distortion[0],
                p2=self.tangential_distortion[1],
            )

        viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
        return viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    def pixels_to_viewdirs(self, pixels: np.ndarray) -> np.ndarray:

        if pixels.shape[-1] != 2:
            raise ValueError("The last dimension of pixels must be 2.")

        batch_shape = pixels.shape[:-1]
        pixels = np.reshape(pixels, (-1, 2))

        local_viewdirs = self.pixels_to_local_viewdirs(pixels)
        viewdirs = matv(self.orientation.T, local_viewdirs)

        # Normalize rays.
        viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = viewdirs.reshape((*batch_shape, 3))
        return viewdirs

    def get_pixels(self, use_center: Optional[bool] = None):
        if use_center is None:
            use_center = self.use_center
        xx, yy = np.meshgrid(
            np.arange(self.image_size_x, dtype=np.float32),
            np.arange(self.image_size_y, dtype=np.float32),
        )
        offset = 0.5 if use_center else 0
        return np.stack([xx, yy], axis=-1) + offset

    def pixels_to_points(
        self,
        pixels: np.ndarray,
        depth: np.ndarray,
        use_projective_depth: Optional[bool] = None,
    ) -> np.ndarray:

        if use_projective_depth is None:
            use_projective_depth = self.use_projective_depth

        rays_through_pixels = self.pixels_to_viewdirs(pixels)
        cosa = 1 if not use_projective_depth else self.pixels_to_cosa(pixels)
        points = rays_through_pixels * depth / cosa + self.position
        return points


    def pixels_to_cosa(self, pixels: np.ndarray) -> np.ndarray:
        rays_through_pixels = self.pixels_to_viewdirs(pixels)
        return (rays_through_pixels @ self.optical_axis)[..., None]

class iPhoneDataset(Dataset):
    def __init__(self, data_root, scene, use_undistort=False):
        super().__init__()
        self.data_root = data_root
        self.scene = scene
        self.use_undistort = use_undistort

        (
            self._center,
            self._scale,
            self._near,
            self._far,
        ) = self._load_scene_info(osp.join(self.data_root, self.scene))

        (
            self._frame_names_map,
            self._time_ids,
            self._camera_ids,
        ) = self._load_metadata_info(osp.join(self.data_root, self.scene))

        arr = self._frame_names_map  # 假设 shape=(N, 3)，dtype='<U7'
        mask = (arr[:, 0] != '') & (arr[:, 1] != '')

        self.valid_index = np.where(mask)[0]

        self._load_extra_info()

    def _load_scene_info(self, data_dir):
        scene_dict = load_json(osp.join(data_dir, "scene.json"))
        center = np.array(scene_dict["center"], dtype=np.float32)
        scale = scene_dict["scale"]
        near = scene_dict["near"]
        far = scene_dict["far"]
        return center, scale, near, far

    def _load_metadata_info(self, data_dir):
        dataset_dict = load_json(osp.join(data_dir, "dataset.json"))
        _frame_names = np.array(dataset_dict["ids"])

        metadata_dict = load_json(osp.join(data_dir, "metadata.json"))
        time_ids = np.array(
            [metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32
        )
        camera_ids = np.array(
            [metadata_dict[k]["camera_id"] for k in _frame_names], dtype=np.uint32
        )

        frame_names_map = np.zeros(
            (time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype
        )
        for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
            frame_names_map[t, c] = _frame_names[i]

        return frame_names_map, time_ids, camera_ids

    def _load_extra_info(self) -> None:
        extra_path = osp.join(self.data_root, self.scene, "extra.json")
        extra_dict = load_json(extra_path)
        self._factor = extra_dict["factor"]
        self._fps = extra_dict["fps"]
        self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
        self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
        self._up = np.array(extra_dict["up"], dtype=np.float32)

    def load_depth(
        self,
        time_id: int,
        camera_id: int,
    ) -> np.ndarray:
        frame_name = self._frame_names_map[time_id, camera_id]
        depth_path = osp.join(
            self.data_root, self.scene, "depth", f"{self._factor}x", frame_name + ".npy"
        )
        depth = np.load(depth_path) * self.scale
        camera = self.load_camera(time_id, camera_id)
        depth = depth / camera.pixels_to_cosa(camera.get_pixels())
        return depth

    def load_camera(
        self,
        time_id: int,
        camera_id: int,
        use_undistort=None,
    ):
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        camera = (
            Camera.fromjson(
                osp.join(self.data_root, self.scene, "camera", frame_name + ".json")
            )
            .rescale_image_domain(1 / self._factor)
            .translate(-self._center)
            .rescale(self._scale)
        )
        if use_undistort:
            camera = camera.undistort_image_domain()
        return camera
    
    def load_rgba(
        self,
        time_id: int,
        camera_id: int,
        *,
        use_undistort: Optional[bool] = None,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        rgb_path = osp.join(
            self.data_root, self.scene,
            "rgb" if not use_undistort else "rgb_undistort",
            f"{self._factor}x",
            frame_name + ".png",
        )
        if osp.exists(rgb_path):
            rgba = load_img(rgb_path, flags=cv2.IMREAD_UNCHANGED)
            if rgba.shape[-1] == 3:
                rgba = np.concatenate(
                    [rgba, np.full_like(rgba[..., :1], 255)], axis=-1
                )
        elif use_undistort:
            camera = self.load_camera(time_id, camera_id, use_undistort=False)
            rgb = self.load_rgba(time_id, camera_id, use_undistort=False)[
                ..., :3
            ]
            rgb = cv2.undistort(rgb, camera.intrin, camera.distortion)
            alpha = (
                cv2.undistort(
                    np.full_like(rgb, 255),
                    camera.intrin,
                    camera.distortion,
                )
                == 255
            )[..., :1].astype(np.uint8) * 255
            rgba = np.concatenate([rgb, alpha], axis=-1)
        else:
            raise ValueError(f"RGB image not found: {rgb_path}.")
        return rgba

    def warp_a2b(self, time_a_id, camera_a_id, time_b_id, camera_b_id):
        camera_a: Camera = self.load_camera(time_id=time_a_id, camera_id=camera_a_id)
        camera_b: Camera = self.load_camera(time_id=time_b_id, camera_id=camera_b_id)

        # get world points of a
        depth_a = self.load_depth(time_a_id, camera_a_id)
        pixel_a = self.load_rgba(time_a_id, camera_a_id) # TODO detete after debug
        pixel_b = self.load_rgba(time_b_id, camera_b_id)

        pixel_a_flat = pixel_a[:,:,:3] 
        world_points_a = camera_a.pixels_to_points(
            pixels=camera_a.get_pixels(),
            depth=depth_a,
        )
        pixels, _ = camera_b.project(points=world_points_a,
                                         return_depth=True)
        # pixels = np.round(pixels[..., ::-1]).astype(np.int32)
        pixels = np.round(pixels).astype(np.int32)

        v_b, u_b = pixels[:,:,0], pixels[:,:,1]

        w = camera_b.image_shape[0]
        h = camera_b.image_shape[1]

        valid = (
            (u_b >= 0) & (u_b < w) &
            (v_b >= 0) & (v_b < h)
        )

        # 应用mask
        u_b = u_b[valid]
        v_b = v_b[valid]
        pixel_src = pixel_a_flat[valid]  # 对应的源像素

        warped_img = np.zeros_like(pixel_a[:,:,:3])
        warped_img[u_b, v_b] = pixel_src

        valid_mask = np.zeros((w, h), dtype=np.uint8)
        valid_mask[u_b, v_b] = 1

        return pixel_a, pixel_b, warped_img, valid_mask

    @property
    def scale(self):
        return self._scale