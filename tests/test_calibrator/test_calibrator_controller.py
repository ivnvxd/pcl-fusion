from unittest.mock import MagicMock, patch

import pytest

from calibrator.calibrator_cam import CalibratorCamera
from calibrator.calibrator_controller import (
    KEY_CALIBRATE,
    KEY_QUIT,
    CalibratorController,
)


@pytest.fixture
def mock_camera():
    mock = MagicMock(spec=CalibratorCamera)
    mock.friendly_id = "mock_friendly_id"
    mock.mxid = "mock_mxid"
    return mock


@patch("cv2.waitKey", return_value=KEY_QUIT)
def test_init_and_quit(mock_waitKey, mock_camera):
    controller = CalibratorController([mock_camera])
    assert controller.running is False


@patch("cv2.waitKey", return_value=KEY_CALIBRATE)
@patch.object(CalibratorController, "_run", return_value=None)  # Mocking _run
def test_init_and_calibrate(mock_run, mock_waitKey, mock_camera):
    controller = CalibratorController([mock_camera])
    controller._handle_key(KEY_CALIBRATE)  # Manually triggering _handle_key
    mock_camera.calibrate.assert_called_once()


@patch("cv2.waitKey", return_value=KEY_QUIT)
def test_update_cameras(mock_waitKey, mock_camera):
    CalibratorController([mock_camera])
    mock_camera.update.assert_called()
