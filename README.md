# Point Cloud Fusion

## Introduction

Point Cloud Fusion is a tool for fusing multiple point clouds into a unified, coherent model. This repository provides an easy-to-use interface for camera calibration and point cloud manipulation.

## Getting Started

### Prerequisites

Ensure you have Python 3.10 installed. The repository uses Open3D, which is compatible specifically with this Python version.

### Installation Steps

1. Clone the GitHub repository:

    ```sh
    git clone git@github.com:voluview/fusion.git
    ```

2. Navigate into the cloned directory and install [Poetry](https://python-poetry.org/docs/#installation):

    ```sh
    cd fusion
    ```

3. Set up the virtual environment and install dependencies:

    ```sh
    poetry env use python3.10
    poetry install
    ```

    > **Note**: Open3D requires Python 3.10; newer versions are not supported.

## Configuration

You can customize the tool's behavior by modifying the settings in `core/config.py`. This file contains various parameters related to camera setup, point cloud alignment, and more.

## How to Use

### Camera Calibration

To achieve accurate point cloud fusion, you'll need to calibrate your cameras first.

#### Steps for Calibration:

1. Generate a ChaRuCo board:
   Utilize the `utils/charuco_board.py` script to generate a ChaRuCo board for calibration.

    ```sh
    make board
    ```

2. Run the calibration script:

    ```sh
    make cal
    ```

#### Controls:

| Key | Action       |
|-----|--------------|
| `q` | Quit         |
| `c` | Calibrate point clouds |

### Generating Point Clouds

After calibrating the cameras, proceed to generate the point clouds.

#### Steps to Generate Point Clouds:

1. Execute the point cloud script:

    ```sh
    make pcl
    ```

#### Controls:

| Key | Action                  |
|-----|-------------------------|
| `q` | Quit                    |
| `a` | Align point clouds with ICP |
| `r` | Reset alignment         |
| `s` | Save point cloud alignment |
| `d` | Toggle depth view       |
