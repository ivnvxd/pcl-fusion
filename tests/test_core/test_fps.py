from unittest.mock import patch

from core.fps import FPSCounter


def test_initialization():
    fps = FPSCounter(duration=5)
    assert fps.duration == 5
    assert fps.frame_timestamps == []


def test_initialization_with_optional_duration():
    fps = FPSCounter()
    assert fps.duration == 5


@patch("time.time")
def test_show_within_duration(mock_time):
    mock_time.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    fps = FPSCounter(duration=5)
    for _ in range(6):
        avg_fps = fps.show()

    # With 6 frames within a duration of 5 seconds, the average FPS should be 6/5 = 1.2
    assert avg_fps == 1.2


@patch("time.time")
def test_show_outside_duration(mock_time):
    mock_time.side_effect = [0.0, 1.0, 7.0]
    fps = FPSCounter(duration=5)

    fps.show()
    fps.show()
    avg_fps = fps.show()

    # Only the last frame is within the 5-second duration, so average FPS should be 1/5 = 0.2
    assert avg_fps == 0.2


@patch("time.time")
def test_show_no_frames(mock_time):
    mock_time.return_value = 0.0
    fps = FPSCounter(duration=5)

    # Clear frame_timestamps to simulate no frames being added
    fps.frame_timestamps.clear()

    avg_fps = fps.show()
    assert avg_fps == 0.2
