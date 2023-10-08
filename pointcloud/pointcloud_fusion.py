from typing import List

import numpy as np
import open3d as o3d
from loguru import logger

import core.config as cfg
from core.fps import FPSCounter
from pointcloud.icp import icp_align
from pointcloud.pointcloud_cam import PointcloudCamera


class PointcloudFusion:
    """
    Class for the point cloud fusion application.
    """

    def __init__(self, pcl_cameras: List[PointcloudCamera]) -> None:
        self.pcl_cameras: List[PointcloudCamera] = pcl_cameras
        self.point_cloud = o3d.geometry.PointCloud()

        self.point_cloud_window: o3d.visualization.VisualizerWithKeyCallback = o3d.visualization.VisualizerWithKeyCallback()  # type: ignore  # noqa: E501
        self.point_cloud_window.register_key_callback(
            ord("A"), lambda vis: self.align_point_clouds()
        )
        self.point_cloud_window.register_key_callback(
            ord("D"), lambda vis: self.toggle_depth()
        )
        self.point_cloud_window.register_key_callback(
            ord("S"), lambda vis: self.save_point_cloud_alignment()
        )
        self.point_cloud_window.register_key_callback(
            ord("R"), lambda vis: self.reset_alignment()
        )
        self.point_cloud_window.register_key_callback(ord("Q"), lambda vis: self.quit())

        self.point_cloud_window.create_window(window_name="Pointcloud")
        self.point_cloud_window.add_geometry(self.point_cloud)

        # Set point size
        opt = self.point_cloud_window.get_render_option()
        opt.point_size = cfg.POINT_SIZE

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, -0.2, 0]
        )
        self.point_cloud_window.add_geometry(origin)
        view = self.point_cloud_window.get_view_control()
        view.set_constant_z_far(cfg.MAX_RANGE * 2)

        self.fps_counter: FPSCounter = FPSCounter(duration=7)

        logger.info("Pointcloud fusion initialised.")

        self.running: bool = True
        self.run()

    def update(self) -> None:
        """
        Update the point cloud by updating each camera and combining the point clouds.
        """

        self.point_cloud.clear()

        for camera in self.pcl_cameras:
            camera.update()
            self.point_cloud += camera.point_cloud

        print("FPS: {:.2f}".format(self.fps_counter.show()), end="\r")

        self.point_cloud_window.update_geometry(self.point_cloud)
        self.point_cloud_window.poll_events()
        self.point_cloud_window.update_renderer()

    def align_point_clouds(self) -> None:
        """
        Fine-tune the alignment of the point clouds.
        """

        transformations: List[np.ndarray] = icp_align(self.pcl_cameras)

        for camera, transformation in zip(self.pcl_cameras[1:], transformations):
            camera.point_cloud_alignment = transformation

        logger.info("Point clouds aligned.")

    def reset_alignment(self) -> None:
        """
        Reset the point cloud alignment for each camera to the identity matrix.
        """

        for camera in self.pcl_cameras:
            camera.point_cloud_alignment = np.identity(4)
            # camera.save_point_cloud_alignment()

    def toggle_depth(self) -> None:
        """
        Toggle the depth visualization on/off.
        """

        for camera in self.pcl_cameras:
            camera.show_depth = not camera.show_depth

    def save_point_cloud_alignment(self) -> None:
        """
        Save the current point cloud alignment to a file.
        """

        for camera in self.pcl_cameras:
            camera.save_point_cloud_alignment()

    def run(self) -> None:
        """
        Run the application.
        """

        while self.running:
            self.update()

    def quit(self) -> None:
        """
        Quit the application.
        """

        self.running = False
