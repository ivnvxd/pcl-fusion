import time
from typing import List


class FPSCounter:
    """
    Calculates the average FPS over a given duration.
    """

    def __init__(self, duration: int = 5) -> None:
        self.frame_timestamps: List[float] = []
        self.duration: int = duration  # seconds

    def show(self) -> float:
        current_time: float = time.time()
        self.frame_timestamps.append(current_time)

        # Remove timestamps older than 'duration' seconds
        while self.frame_timestamps and (
            current_time - self.frame_timestamps[0] > self.duration
        ):
            self.frame_timestamps.pop(0)

        # Calculate average FPS over the remaining timestamps
        num_frames: int = len(self.frame_timestamps)
        avg_fps: float = num_frames / self.duration if self.frame_timestamps else 0

        return avg_fps
