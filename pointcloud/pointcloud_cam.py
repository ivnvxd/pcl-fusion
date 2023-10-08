from typing import Optional

import cv2
import depthai as dai
import numpy as np
import open3d as o3d
from loguru import logger

import core.config as cfg
from pointcloud.host_sync import HostSync
from pointcloud.make_pointcloud import rgbd_to_point_cloud


class PointcloudCamera:
    """
    A class for managing a DepthAI camera instance.
    """

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int) -> None:
        self.device_info: dai.DeviceInfo = device_info
        self.friendly_id: int = friendly_id
        self.mxid: str = device_info.getMxId()

        # load default settings
        self.show_video: bool = cfg.SHOW_VIDEO
        self.show_point_cloud: bool = cfg.SHOW_POINT_CLOUD
        self.show_depth: bool = False
        self.image_frame: Optional[np.ndarray] = None
        self.depth_frame: Optional[np.ndarray] = None
        self.depth_visualization_frame: Optional[np.ndarray] = None
        self.point_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()

        # create pipeline
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.device.setIrLaserDotProjectorBrightness(cfg.LASER_DOT)
        self.device.setIrFloodLightBrightness(cfg.FLOOD_LIGHT)

        self.image_queue: dai.DataOutputQueue = self.device.getOutputQueue(
            name="image", maxSize=10, blocking=False
        )
        self.depth_queue: dai.DataOutputQueue = self.device.getOutputQueue(
            name="depth", maxSize=10, blocking=False
        )
        self.host_sync: HostSync = HostSync(["image", "depth"])

        self._create_windows()
        self._load_calibration()

        logger.info(f"{self.friendly_id}: Connected to {self.mxid}")

    def _create_windows(self) -> None:
        """Creates the windows for displaying the camera and point cloud frames."""

        # camera window
        self.window_name: str = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        if self.show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 360)

        # point cloud window
        if self.show_point_cloud:
            self.point_cloud_window = o3d.visualization.Visualizer()  # type: ignore
            self.point_cloud_window.create_window(
                window_name=f"[{self.friendly_id}] Point Cloud - mxid: {self.mxid}"
            )
            self.point_cloud_window.add_geometry(self.point_cloud)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            self.point_cloud_window.add_geometry(origin)
            self.point_cloud_window.get_view_control().set_constant_z_far(
                cfg.MAX_RANGE * 2
            )

    def _load_calibration(self) -> None:
        """
        Loads the camera extrinsics, intrinsics, and point cloud alignment data from the calibration data directory.
        """  # noqa: E501

        self._load_extrinsics()
        self._load_intrinsics()
        self._load_point_cloud_alignment()

        print(self.pinhole_camera_intrinsic)

    def _load_extrinsics(self):
        """
        Loads the camera extrinsics from a numpy file in the calibration data directory.
        """

        path = f"{cfg.CALIBRATION_DATA_DIR}/extrinsics_{self.mxid}.npz"

        try:
            extrinsics = np.load(path)
            self.cam_to_world = extrinsics["cam_to_world"]
            self.world_to_cam = extrinsics["world_to_cam"]

            logger.info(
                f"{self.friendly_id}: Loaded calibration data for camera {self.mxid} from {path}"  # noqa: E501
            )

        except FileNotFoundError:
            logger.warning(
                f"{self.friendly_id}: Could not load calibration data for camera {self.mxid} from {path}!"  # noqa: E501
            )

            self.cam_to_world = np.eye(4)
            self.world_to_cam = np.eye(4)

    def _load_intrinsics(self) -> None:
        """
        Loads the camera intrinsics from the device.
        """

        calibration = self.device.readCalibration()
        self.intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB if cfg.COLOR else dai.CameraBoardSocket.RIGHT,
            dai.Size2f(*self.image_size),
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size,
            self.intrinsics[0][0],
            self.intrinsics[1][1],
            self.intrinsics[0][2],
            self.intrinsics[1][2],
        )

        logger.info(
            f"{self.friendly_id}: Loaded intrinsics for camera {self.mxid} from device"
        )

    def _load_point_cloud_alignment(self) -> None:
        """
        Loads the point cloud alignment data from a numpy file in the calibration data directory.
        """  # noqa: E501

        path = f"{cfg.CALIBRATION_DATA_DIR}/point_cloud_alignment_{self.mxid}.npy"

        try:
            self.point_cloud_alignment = np.load(path)

            logger.info(
                f"{self.friendly_id}: Loaded point cloud alignment for camera {self.mxid} from {path}"  # noqa: E501
            )
        except FileNotFoundError:
            logger.warning(
                f"{self.friendly_id}: Could not load point cloud alignment for camera {self.mxid} from {path}!"  # noqa: E501
            )

            self.point_cloud_alignment = np.eye(4)

    def _create_pipeline(self) -> None:
        """
        Creates a DepthAI pipeline for capturing stereo images and generating depth maps.
        """  # noqa: E501

        pipeline = dai.Pipeline()

        # Depth cam -> 'depth'
        mono_left = pipeline.createMonoCamera()
        mono_right = pipeline.createMonoCamera()
        mono_left.setResolution(cfg.MONO_RESOLUTION)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(cfg.MONO_RESOLUTION)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        cam_stereo = pipeline.createStereoDepth()
        cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        cam_stereo.initialConfig.setMedianFilter(cfg.MEDIAN)
        cam_stereo.initialConfig.setConfidenceThreshold(cfg.CONFIDENCE_THRESHOLD)
        cam_stereo.setLeftRightCheck(cfg.LRCHECK)
        cam_stereo.setExtendedDisparity(cfg.EXTENDED)
        cam_stereo.setSubpixel(cfg.SUBPIXEL)

        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)

        init_config = cam_stereo.initialConfig.get()
        init_config.postProcessing.speckleFilter.enable = cfg.SPECKLE
        init_config.postProcessing.speckleFilter.speckleRange = 50
        init_config.postProcessing.temporalFilter.enable = cfg.TEMPORAL
        init_config.postProcessing.spatialFilter.enable = cfg.SPATIAL
        init_config.postProcessing.spatialFilter.holeFillingRadius = 2
        init_config.postProcessing.spatialFilter.numIterations = 1
        init_config.postProcessing.thresholdFilter.minRange = cfg.MIN_RANGE
        init_config.postProcessing.thresholdFilter.maxRange = cfg.MAX_RANGE
        init_config.postProcessing.decimationFilter.decimationFactor = 1
        cam_stereo.initialConfig.set(init_config)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        cam_stereo.depth.link(xout_depth.input)

        # RGB cam or mono right -> 'image'
        xout_image = pipeline.createXLinkOut()
        xout_image.setStreamName("image")
        if cfg.COLOR:
            cam_rgb = pipeline.createColorCamera()
            cam_rgb.setResolution(cfg.RGB_RESOLUTION)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            cam_rgb.setIspScale(1, cfg.ISP_SCALE)

            if cfg.MANUAL_EXPOSURE:
                cam_rgb.initialControl.setManualExposure(cfg.EXPOSURE, cfg.ISO)
            if cfg.MANUAL_FOCUS:
                cam_rgb.initialControl.setManualFocus(cfg.FOCUS)
            if cfg.MANUAL_WHITEBALANCE:
                cam_rgb.initialControl.setManualWhiteBalance(cfg.WHITEBALANCE)

            cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            cam_rgb.isp.link(xout_image.input)
            self.image_size = cam_rgb.getIspSize()
        else:
            cam_stereo.rectifiedRight.link(xout_image.input)
            self.image_size = mono_right.getResolutionSize()

        self.pipeline = pipeline

    def update(self) -> None:
        """
        Update the camera's depth and image frames, and display them if `show_video` is True.
        Convert the image frame to RGB and generate a point cloud from the depth and RGB frames.
        Display the point cloud if `show_point_cloud` is True.
        """  # noqa: E501

        for queue in [self.depth_queue, self.image_queue]:
            new_msgs = queue.tryGetAll()
            if new_msgs is not None:
                for new_msg in new_msgs:
                    self.host_sync.add(queue.getName(), new_msg)

        msg_sync = self.host_sync.get()
        if msg_sync is None:
            return

        self.depth_frame = msg_sync["depth"].getFrame()
        self.image_frame = msg_sync["image"].getCvFrame()

        self.depth_visualization_frame = cv2.normalize(
            self.depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1  # type: ignore
        )
        self.depth_visualization_frame = cv2.equalizeHist(
            self.depth_visualization_frame
        )
        self.depth_visualization_frame = cv2.applyColorMap(
            self.depth_visualization_frame, cv2.COLORMAP_HOT
        )

        if self.show_video:
            if self.show_depth:
                cv2.imshow(self.window_name, self.depth_visualization_frame)
            else:
                cv2.imshow(self.window_name, self.image_frame)
            cv2.waitKey(1)

        rgb_image = cv2.cvtColor(self.image_frame, cv2.COLOR_BGR2RGB)
        # self.rgbd_to_point_cloud(self.depth_frame, rgb_image)

        self.point_cloud = rgbd_to_point_cloud(
            self.depth_frame,
            rgb_image,
            self.pinhole_camera_intrinsic,
            self.world_to_cam,
        )
        self.point_cloud.transform(self.point_cloud_alignment)

        if self.show_point_cloud:
            self.point_cloud_window.update_geometry(self.point_cloud)
            self.point_cloud_window.poll_events()
            self.point_cloud_window.update_renderer()

    def save_point_cloud_alignment(self) -> None:
        """
        Saves the point cloud alignment data to a numpy file in the calibration data directory.
        The file name is generated based on the mxid of the camera instance.
        """  # noqa: E501

        np.save(
            f"{cfg.CALIBRATION_DATA_DIR}/point_cloud_alignment_{self.mxid}.npy",
            self.point_cloud_alignment,
        )

    def __del__(self) -> None:
        """
        Closes the camera instance.
        """

        self.device.close()
        logger.info(f"{self.friendly_id}: Closed {self.mxid}")
