from unittest.mock import MagicMock, patch

import numpy as np
import open3d as o3d
import pytest

from pointcloud.pointcloud_fusion import PointcloudFusion


# Mock for PointcloudCamera class
class MockPointcloudCamera:
    def __init__(self):
        self.point_cloud = MagicMock()
        self.point_cloud_alignment = MagicMock()
        self.show_depth = False
        self.update = MagicMock()
        self.save_point_cloud_alignment = MagicMock()


@pytest.fixture
def mock_cameras():
    return [MockPointcloudCamera() for _ in range(3)]


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
@patch("pointcloud.pointcloud_fusion.o3d.geometry.PointCloud")
@patch("pointcloud.pointcloud_fusion.o3d.visualization.VisualizerWithKeyCallback")
def test_init(mock_VisualizerWithKeyCallback, mock_PointCloud, mock_cameras):
    pointcloud_fusion = PointcloudFusion(mock_cameras)
    assert pointcloud_fusion.point_cloud == mock_PointCloud.return_value


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
@patch("pointcloud.pointcloud_fusion.PointcloudCamera")
def test_update(mock_run, mock_camera_class):
    mock_camera = MagicMock()
    mock_camera.point_cloud = o3d.geometry.PointCloud()
    mock_camera_class.return_value = mock_camera

    pointcloud_fusion = PointcloudFusion([mock_camera])
    pointcloud_fusion.point_cloud_window = MagicMock(
        spec=o3d.visualization.VisualizerWithKeyCallback
    )

    pointcloud_fusion.update()
    pointcloud_fusion.point_cloud_window.update_geometry.assert_called()


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
@patch("pointcloud.pointcloud_fusion.PointcloudCamera")
def test_align_point_clouds(mock_run, mock_camera_class):
    # Prepare Mock Cameras
    mock_camera = MagicMock()
    mock_camera.point_cloud = MagicMock(spec=o3d.geometry.PointCloud)
    mock_camera.point_cloud_alignment = np.identity(4)

    mock_master_camera = MagicMock()
    mock_master_camera.point_cloud = MagicMock(spec=o3d.geometry.PointCloud)
    mock_master_camera.point_cloud_alignment = np.identity(4)

    # Initialize PointcloudFusion class
    pointcloud_fusion = PointcloudFusion([mock_master_camera, mock_camera])

    # Mock external methods that would otherwise perform complex operations
    mock_master_camera.point_cloud.voxel_down_sample = MagicMock()
    mock_master_camera.point_cloud.estimate_normals = MagicMock()
    mock_camera.point_cloud.voxel_down_sample = MagicMock()
    mock_camera.point_cloud.estimate_normals = MagicMock()

    # Mock ICP Registration Method
    mock_icp_result = MagicMock()
    mock_icp_result.transformation = np.identity(4)
    with patch(
        "pointcloud.pointcloud_fusion.o3d.pipelines.registration.registration_colored_icp"
    ) as mock_registration_colored_icp:
        mock_registration_colored_icp.return_value = mock_icp_result

        # Call the method
        pointcloud_fusion.align_point_clouds()

    # Validate the calls
    mock_master_camera.point_cloud.voxel_down_sample.assert_called()
    # mock_master_camera.point_cloud.estimate_normals.assert_called()
    mock_camera.point_cloud.voxel_down_sample.assert_called()
    # mock_camera.point_cloud.estimate_normals.assert_called()
    mock_registration_colored_icp.assert_called()
    assert np.array_equal(
        mock_camera.point_cloud_alignment, mock_icp_result.transformation
    )


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
def test_reset_alignment(mock_run, mock_cameras):
    pointcloud_fusion = PointcloudFusion(mock_cameras)
    pointcloud_fusion.reset_alignment()

    for camera in mock_cameras:
        assert np.array_equal(camera.point_cloud_alignment, np.identity(4))


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
def test_toggle_depth(mock_run, mock_cameras):
    pointcloud_fusion = PointcloudFusion(mock_cameras)
    pointcloud_fusion.toggle_depth()

    for camera in mock_cameras:
        assert camera.show_depth is True


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
def test_save_point_cloud_alignment(mock_run, mock_cameras):
    pointcloud_fusion = PointcloudFusion(mock_cameras)
    pointcloud_fusion.save_point_cloud_alignment()

    for camera in mock_cameras:
        camera.save_point_cloud_alignment.assert_called_once()


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
def test_run(mock_run, mock_cameras):
    pointcloud_fusion = PointcloudFusion(mock_cameras)
    pointcloud_fusion.running = False
    pointcloud_fusion.run()

    # TODO: Add assertions based on expected behavior


@patch("pointcloud.pointcloud_fusion.PointcloudFusion.run")
def test_quit(mock_run, mock_cameras):
    pointcloud_fusion = PointcloudFusion(mock_cameras)
    pointcloud_fusion.quit()

    assert pointcloud_fusion.running is False
