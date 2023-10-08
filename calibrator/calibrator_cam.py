import os
import time

import cv2
import depthai as dai
import numpy as np
from loguru import logger

import core.config as cfg


class CalibratorCamera:
    """
    Class for calibrating a camera.
    """

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int):
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()

        # create pipeline
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(
            name="rgb", maxSize=1, blocking=False
        )
        self.still_queue = self.device.getOutputQueue(
            name="still", maxSize=1, blocking=False
        )
        self.control_queue = self.device.getInputQueue(name="control")

        self._create_windows()
        self._load_calibration()
        self._load_calibration_parameters()

        logger.info(f"{self.friendly_id}: Connected to {self.mxid}")

    def _create_windows(self) -> None:
        """
        Creates a named window for the camera.
        """

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 360)

    def _load_calibration(self) -> None:
        """
        Loads the camera calibration parameters.
        """

        # Camera intrinsic parameters
        self.intrinsic_mat = np.array(
            self.device.readCalibration().getCameraIntrinsics(
                dai.CameraBoardSocket.RGB, 3840, 2160
            )
        )
        self.distortion_coef = np.zeros((1, 5))

        # Camera extrinsic parameters
        self.rot_vec = None
        self.trans_vec = None
        self.world_to_cam = None
        self.cam_to_world = None

    def _load_calibration_parameters(self) -> None:
        """
        Loads the calibration parameters for the camera.
        """

        self.calibration_data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), cfg.CALIBRATION_DATA_DIR
        )

        if cfg.CHARUCO_CALIBRATION:
            # Charuco board parameters
            checkerboard_size = cfg.CHECKERBOARD_SIZE
            square_length = cfg.SQUARE_LENGTH
            marker_length = cfg.MARKER_LENGTH
            aruco_dict = cfg.ARUCO_DICT

            board = cv2.aruco.CharucoBoard(
                size=checkerboard_size,
                squareLength=square_length,
                markerLength=marker_length,
                dictionary=aruco_dict,
            )

            self.corners_world = board.getChessboardCorners()
            self.charuco_detector = cv2.aruco.CharucoDetector(board)

        else:
            # Checkerboard parameters
            checkerboard_size = cfg.CHECKERBOARD_SIZE
            self.checkerboard_inner_size = (
                checkerboard_size[0] - 1,
                checkerboard_size[1] - 1,
            )
            square_size = cfg.SQUARE_LENGTH
            self.corners_world = np.zeros(
                (
                    1,
                    self.checkerboard_inner_size[0] * self.checkerboard_inner_size[1],
                    3,
                ),
                np.float32,
            )
            self.corners_world[0, :, :2] = np.mgrid[
                0 : self.checkerboard_inner_size[0],
                0 : self.checkerboard_inner_size[1],
            ].T.reshape(-1, 2)
            self.corners_world *= square_size

    def _create_pipeline(self) -> None:
        """
        Creates the DepthAI pipeline for the camera.
        """

        pipeline = dai.Pipeline()

        # RGB cam -> 'rgb'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(640, 360)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # Still encoder -> 'still'
        still_encoder = pipeline.create(dai.node.VideoEncoder)
        still_encoder.setDefaultProfilePreset(
            1, dai.VideoEncoderProperties.Profile.MJPEG
        )
        cam_rgb.still.link(still_encoder.input)
        xout_still = pipeline.createXLinkOut()
        xout_still.setStreamName("still")
        still_encoder.bitstream.link(xout_still.input)

        # Camera control -> 'control'
        control = pipeline.create(dai.node.XLinkIn)
        control.setStreamName("control")
        control.out.link(cam_rgb.inputControl)

        self.pipeline = pipeline

    def update(self) -> None:
        """
        Update the camera and display the RGB frame.
        """

        in_rgb = self.rgb_queue.tryGet()

        if in_rgb is None:
            return

        self.frame_rgb = in_rgb.getCvFrame()  # type: ignore

        cv2.imshow(self.window_name, self.frame_rgb)

    def calibrate(self) -> None:
        """
        Calibrate the camera.
        """

        frame_rgb = self.capture_still()
        if frame_rgb is None:
            logger.error(
                f"{self.friendly_id}: Could not capture still image {self.mxid}"
            )
            return

        if cfg.CHARUCO_CALIBRATION:
            pose: dict = self.estimate_pose_charuco(frame_rgb)

            cv2.aruco.drawDetectedMarkers(
                frame_rgb, pose["marker_corners"], pose["marker_ids"]
            )
        else:
            pose: dict = None

        if pose is None:
            logger.error(f"{self.friendly_id}: Could not estimate pose {self.mxid}")
            return

        reprojection: np.ndarray = self.draw_origin(frame_rgb, pose)

        # Save the calibration data and image
        self.save_calibration(pose, reprojection)

        # Show the image with the detected markers and origin
        cv2.imshow(self.window_name, reprojection)

        logger.info(f"{self.friendly_id}: Calibration complete {self.mxid}")

    def capture_still(self, timeout_ms: int = 1000) -> np.ndarray:
        """
        Capture a still high-resolution image from the camera.

        Args:
            timeout_ms (int, optional): Timeout in milliseconds. Defaults to 1000.

        Returns:
            np.ndarray: The captured image.
        """

        logger.info(f"{self.friendly_id}: Capturing still image... {self.mxid}")

        # Empty the queue
        self.still_queue.tryGetAll()

        # Send a capture command
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        self.control_queue.send(ctrl)

        # Wait for the still to be captured
        in_still = None
        start_time = time.time() * 1000
        while in_still is None:
            time.sleep(0.1)
            in_still = self.still_queue.tryGet()
            if time.time() * 1000 - start_time > timeout_ms:
                logger.warning(
                    f"{self.friendly_id}: Did not recieve still image - retrying... {self.mxid}"  # noqa: E501
                )
                return self.capture_still(timeout_ms)

        still_rgb = cv2.imdecode(in_still.getData(), cv2.IMREAD_UNCHANGED)  # type: ignore  # noqa: E501

        return still_rgb

    def estimate_pose_charuco(self, image: np.ndarray) -> dict:
        """
        Estimates the pose of the camera using a Charuco board.

        Args:
            image (np.ndarray): The RGB frame from the camera.

        Returns:
            pose (dict): The pose of the camera.
        """

        if image is None:
            return  # type: ignore

        logger.info(f"{self.friendly_id}: Estimating pose... {self.mxid}")

        frame_gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect Charuco markers
        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = self.charuco_detector.detectBoard(frame_gray)

        if charuco_corners is None or charuco_ids is None:
            logger.error(f"{self.friendly_id}: Charuco board not found {self.mxid}")
            return  # type: ignore

        # Sort the object points and image points based on charuco_ids
        object_points = np.array(
            [self.corners_world[i] for i in charuco_ids.flatten()], dtype=np.float32
        )
        image_points = np.array(charuco_corners, dtype=np.float32)

        # Refine the corner locations
        image_points = cv2.cornerSubPix(
            frame_gray,
            image_points,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        intrinsic_mat = self.intrinsic_mat
        distortion_coef = self.distortion_coef

        # Estimate the pose of the Charuco board
        _, rot_vec, trans_vec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsic_mat,
            distortion_coef,
        )

        # compute transformation from world to camera space and wise versa
        rot_m = cv2.Rodrigues(rot_vec)[0]
        world_to_cam = np.vstack(
            (np.hstack((rot_m, trans_vec)), np.array([0, 0, 0, 1]))
        )
        cam_to_world = np.linalg.inv(world_to_cam)  # type: ignore

        pose = {
            "world_to_cam": world_to_cam,
            "cam_to_world": cam_to_world,
            "trans_vec": trans_vec,
            "rot_vec": rot_vec,
            "intrinsics": intrinsic_mat,
            "distortion": distortion_coef,
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
        }

        return pose

    def estimate_pose_checkerboard(self, image: np.ndarray) -> dict:
        """
        Estimates the pose of the camera using a checkerboard.

        Args:
            image (np.ndarray): The RGB frame from the camera.

        Returns:
            pose (dict): The pose of the camera.
        """

        if image is None:
            return  # type: ignore

        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the checkerboard corners
        found, corners = cv2.findChessboardCorners(
            frame_gray,
            self.checkerboard_inner_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,  # type: ignore
        )

        if not found:
            return  # type: ignore

        # refine the corner locations
        corners = cv2.cornerSubPix(
            frame_gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        intrinsic_mat = self.intrinsic_mat
        distortion_coef = self.distortion_coef

        # compute the rotation and translation from the camera to the checkerboard
        _, rot_vec, trans_vec = cv2.solvePnP(
            self.corners_world,  # type: ignore
            corners,
            intrinsic_mat,
            distortion_coef,
        )

        # compute transformation from world to camera space and wise versa
        rot_m = cv2.Rodrigues(rot_vec)[0]
        world_to_cam = np.vstack(
            (np.hstack((rot_m, trans_vec)), np.array([0, 0, 0, 1]))
        )
        cam_to_world = np.linalg.inv(world_to_cam)  # type: ignore

        pose = {
            "world_to_cam": world_to_cam,
            "cam_to_world": cam_to_world,
            "trans_vec": trans_vec,
            "rot_vec": rot_vec,
            "intrinsics": intrinsic_mat,
            "distortion": distortion_coef,
        }

        return pose

    def draw_origin(self, image: np.ndarray, pose: dict) -> np.ndarray:
        """
        Draws the origin on the image.

        Args:
            pose (dict): The pose of the camera.

        Returns:
            reprojection (np.ndarray): The image with the origin drawn on it.
        """

        # Define the 3D points of the coordinate system
        points = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]])  # type: ignore  # noqa: E501

        # Project the 3D points onto the 2D image plane using the camera pose
        projected_points, _ = cv2.projectPoints(
            points,  # type: ignore
            pose["rot_vec"],
            pose["trans_vec"],
            pose["intrinsics"],
            pose["distortion"],
        )

        # Draw the coordinate system on the image
        reprojection = image.copy()
        [p_0, p_x, p_y, p_z] = projected_points.astype(np.int64)
        reprojection = cv2.line(
            reprojection, tuple(p_0[0]), tuple(p_x[0]), (0, 0, 255), 2
        )
        reprojection = cv2.line(
            reprojection, tuple(p_0[0]), tuple(p_y[0]), (0, 255, 0), 2
        )
        reprojection = cv2.line(
            reprojection, tuple(p_0[0]), tuple(p_z[0]), (255, 0, 0), 2
        )

        return reprojection

    def save_calibration(self, pose: dict, reprojection: np.ndarray) -> None:
        """
        Save the calibration data.

        Args:
            pose (dict): The pose of the camera.
            reprojection (np.ndarray): The image with the origin drawn on it.
        """

        os.makedirs(self.calibration_data_path, exist_ok=True)

        try:
            np.savez(
                os.path.join(self.calibration_data_path, f"extrinsics_{self.mxid}.npz"),
                world_to_cam=pose["world_to_cam"],
                cam_to_world=pose["cam_to_world"],
                trans_vec=pose["trans_vec"],
                rot_vec=pose["rot_vec"],
            )
        except Exception:
            logger.error(
                f"{self.friendly_id}: Could not save calibration data {self.mxid}"
            )

        image_path = os.path.join(self.calibration_data_path, "img")
        os.makedirs(image_path, exist_ok=True)

        try:
            cv2.imwrite(
                os.path.join(image_path, f"camera_{self.mxid}.png"),
                reprojection,
            )
        except Exception:
            logger.error(
                f"{self.friendly_id}: Could not save calibration image {self.mxid}"
            )

    def __del__(self) -> None:
        """
        Close the device when the object is deleted.
        """

        self.device.close()
        logger.info(f"{self.friendly_id}: Closed {self.mxid}")
