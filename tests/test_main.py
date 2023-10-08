from unittest.mock import Mock, patch

import pytest

from main import main


@pytest.fixture
def mock_device_info():
    return Mock()


@pytest.mark.parametrize(
    "pointcloud,calibrator", [(True, False), (False, True), (False, False)]
)
def test_main(pointcloud, calibrator, mock_device_info):
    with patch("main.get_devices") as mock_get_dev, patch(
        "main.init_pointcloud_cameras"
    ) as mock_init_pcl, patch("main.PointcloudFusion") as mock_pcf, patch(
        "main.init_calibrator_cameras"
    ) as mock_init_cal, patch(
        "main.CalibratorController"
    ) as mock_cc, patch(
        "argparse.ArgumentParser.parse_args"
    ) as mock_args, patch(
        "builtins.print"
    ) as mock_print:
        mock_args.return_value = Mock(pointcloud=pointcloud, calibrator=calibrator)
        mock_get_dev.return_value = [mock_device_info]

        main()

        if pointcloud:
            mock_get_dev.assert_called_once()
            mock_init_pcl.assert_called_once_with([mock_device_info])
            mock_pcf.assert_called_once()
        elif calibrator:
            mock_get_dev.assert_called_once()
            mock_init_cal.assert_called_once_with([mock_device_info])
            mock_cc.assert_called_once()
        else:
            mock_print.assert_called_with(
                "Please specify either -p/--pointcloud or -c/--calibrator"
            )
