import argparse
from typing import List

import depthai as dai

from calibrator.calibrator_cam import CalibratorCamera
from calibrator.calibrator_controller import CalibratorController
from core.devices import get_devices, init_calibrator_cameras, init_pointcloud_cameras
from pointcloud.pointcloud_cam import PointcloudCamera
from pointcloud.pointcloud_fusion import PointcloudFusion


def main() -> None:
    """
    Main function. Parses arguments and runs calibrator or pointcloud.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pointcloud", action="store_true", help="Run pointcloud"
    )
    parser.add_argument(
        "-c", "--calibrator", action="store_true", help="Run calibrator"
    )
    args = parser.parse_args()

    if args.pointcloud:
        device_infos: List[dai.DeviceInfo] = get_devices()
        pcl_cameras: List[PointcloudCamera] = init_pointcloud_cameras(device_infos)
        PointcloudFusion(pcl_cameras)
    elif args.calibrator:
        device_infos: List[dai.DeviceInfo] = get_devices()
        cal_cameras: List[CalibratorCamera] = init_calibrator_cameras(device_infos)
        CalibratorController(cal_cameras)
    else:
        print("Please specify either -p/--pointcloud or -c/--calibrator")


if __name__ == "__main__":
    main()
