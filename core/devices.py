from typing import List

import depthai as dai
from loguru import logger

from calibrator.calibrator_cam import CalibratorCamera
from pointcloud.pointcloud_cam import PointcloudCamera


def get_devices() -> List[dai.DeviceInfo]:
    """
    Get a list of all connected devices.

    Returns:
        A list of all connected devices.
    """

    device_infos: List[dai.DeviceInfo] = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0:
        logger.exception("No devices found!")
        raise RuntimeError("No devices found!")
    else:
        logger.info(f"Found {len(device_infos)} devices")

    device_infos.sort(key=lambda x: x.getMxId(), reverse=True)

    return device_infos


def init_pointcloud_cameras(
    device_infos: List[dai.DeviceInfo],
) -> List[PointcloudCamera]:
    """
    Initialise all connected devices as pointcloud cameras.

    Args:
        device_infos: List of connected device infos.

    Returns:
        List of pointcloud cameras.
    """

    pcl_cameras: List[PointcloudCamera] = []

    friendly_id: int = 1

    for device_info in device_infos:
        camera: PointcloudCamera = PointcloudCamera(device_info, friendly_id)

        pcl_cameras.append(camera)
        friendly_id += 1

    return pcl_cameras


def init_calibrator_cameras(
    device_infos: List[dai.DeviceInfo],
) -> List[CalibratorCamera]:
    """
    Initialise all connected devices as calibrator cameras.

    Args:
        device_infos: List of connected device infos.

    Returns:
        List of calibrator cameras.
    """

    calibrators: List[CalibratorCamera] = []

    friendly_id: int = 1

    for device_info in device_infos:
        camera: CalibratorCamera = CalibratorCamera(device_info, friendly_id)

        calibrators.append(camera)
        friendly_id += 1

    return calibrators
