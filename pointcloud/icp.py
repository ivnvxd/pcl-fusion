from typing import List

import numpy as np
import open3d as o3d

import core.config as cfg
from pointcloud.pointcloud_cam import PointcloudCamera


def icp_align(pcl_cameras: List[PointcloudCamera]) -> List[np.ndarray]:
    """
    Align the point clouds from each camera using ICP (Iterative Closest Point)

    Args:
        pcl_cameras (List[PointcloudCamera]): List of point cloud cameras

    Returns:
        List[np.ndarray]: List of transformation matrices
    """

    voxel_radius: List[float] = cfg.VOXEL_RADIUS
    max_iter: List[int] = cfg.ICP_MAX_ITER

    transformations: List[np.ndarray] = []

    master_point_cloud: o3d.geometry.PointCloud = pcl_cameras[0].point_cloud

    for camera in pcl_cameras[1:]:
        for iter, radius in zip(max_iter, voxel_radius):
            target_down: o3d.geometry.PointCloud = master_point_cloud.voxel_down_sample(
                radius
            )
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )

            source_down: o3d.geometry.PointCloud = camera.point_cloud.voxel_down_sample(
                radius
            )
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
            )

            result_icp: o3d.pipelines.registration.RegistrationResult = (
                o3d.pipelines.registration.registration_colored_icp(
                    source_down,
                    target_down,
                    radius,
                    camera.point_cloud_alignment,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
                    ),
                )
            )

            transformations.append(result_icp.transformation)

    return transformations
