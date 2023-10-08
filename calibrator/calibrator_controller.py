from typing import List

import cv2
from loguru import logger

from calibrator.calibrator_cam import CalibratorCamera

# Constants
KEY_QUIT = ord("q")
KEY_CALIBRATE = ord("c")


class CalibratorController:
    """
    Controller for the calibrator.
    """

    def __init__(self, cal_cameras: List[CalibratorCamera]) -> None:
        self.cal_cameras: List[CalibratorCamera] = cal_cameras
        self.running: bool = True

        logger.info("Calibrator initialized.")
        self._run()

    def _run(self) -> None:
        """
        Main loop for the calibrator.
        """

        while self.running:
            key: int = cv2.waitKey(1)
            self._handle_key(key)
            self._update_cameras()

    def _handle_key(self, key: int) -> None:
        """
        Handles key events.
        """

        if key == KEY_QUIT:
            self._quit()

        elif key == KEY_CALIBRATE:
            self._calibrate_cameras()

    def _quit(self) -> None:
        """
        Sets the running flag to false to quit the application.
        """

        self.running = False
        logger.info("Quitting calibrator...")

    def _calibrate_cameras(self) -> None:
        """
        Calibrates all cameras.
        """

        logger.info("Calibrating cameras...")
        for camera in self.cal_cameras:
            logger.info(f"{camera.friendly_id}: Calibrating camera... {camera.mxid}")
            camera.calibrate()
        cv2.waitKey()

    def _update_cameras(self) -> None:
        """
        Updates all cameras.
        """

        for camera in self.cal_cameras:
            camera.update()
