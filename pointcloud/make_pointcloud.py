from typing import Optional

import numpy as np
import open3d as o3d

import core.config as cfg


def rgbd_to_point_cloud(
    depth_frame: np.ndarray,
    image_frame: np.ndarray,
    pinhole_camera_intrinsic: o3d.camera.PinholeCameraIntrinsic,
    world_to_cam: Optional[np.ndarray],
) -> o3d.geometry.PointCloud:
    """
    Converts an RGB-D image to a point cloud using the camera intrinsic parameters and the world-to-camera transform.

    Args:
        depth_frame (np.ndarray): The depth image as a numpy array.
        image_frame (np.ndarray): The RGB image as a numpy array.
    """  # noqa: E501

    rgb_o3d: o3d.geometry.Image = o3d.geometry.Image(image_frame)
    df: np.ndarray = np.copy(depth_frame).astype(np.float32)
    depth_o3d: o3d.geometry.Image = o3d.geometry.Image(df)
    rgbd_image: o3d.geometry.RGBDImage = (
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            convert_rgb_to_intensity=(len(image_frame.shape) != 3),
        )
    )

    point_cloud: o3d.geometry.PointCloud = (
        o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, pinhole_camera_intrinsic, world_to_cam
        )
    )

    if cfg.DOWNSAMPLE:
        point_cloud = point_cloud.voxel_down_sample(voxel_size=cfg.VOXEL_SIZE)

    if cfg.REMOVE_NOISE:
        point_cloud = point_cloud.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=0.1
        )[0]

    T: np.ndarray = np.eye(4)
    T[1, 1] = -1  # flip y axis
    T[2, 2] = -1  # correct upside down z axis
    point_cloud.transform(T)

    return point_cloud
