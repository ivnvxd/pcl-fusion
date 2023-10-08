from typing import List
from unittest.mock import MagicMock, patch

import pytest

from core.devices import get_devices, init_calibrator_cameras, init_pointcloud_cameras


# Mocking dai.DeviceInfo class
class MockDeviceInfo:
    def __init__(self, mxid):
        self.mxid = mxid

    def getMxId(self):
        return self.mxid


@pytest.fixture
def mock_device_list() -> List[MockDeviceInfo]:
    return [MockDeviceInfo(i) for i in range(3)]


# Test for get_devices
@patch("core.devices.dai.Device.getAllAvailableDevices")
def test_get_devices(mock_get_all_devices, mock_device_list):
    mock_get_all_devices.return_value = mock_device_list

    result = get_devices()

    assert len(result) == 3
    assert result[0].getMxId() == 2  # Should be sorted in reverse order


# Test for init_pointcloud_cameras
@patch("core.devices.PointcloudCamera")
def test_init_pointcloud_cameras(mock_pointcloud_camera, mock_device_list):
    mock_pointcloud_camera.return_value = MagicMock()

    result = init_pointcloud_cameras(mock_device_list)

    assert len(result) == 3
    mock_pointcloud_camera.assert_called()


# Test for init_calibrator_cameras
@patch("core.devices.CalibratorCamera")
def test_init_calibrator_cameras(mock_calibrator_camera, mock_device_list):
    mock_calibrator_camera.return_value = MagicMock()

    result = init_calibrator_cameras(mock_device_list)

    assert len(result) == 3
    mock_calibrator_camera.assert_called()
