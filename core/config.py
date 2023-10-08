import cv2
import depthai as dai

# CAMERA CONFIG
LASER_DOT = 800  # Laser dot projector brightness, in mA, 0..1200
FLOOD_LIGHT = 300  # Flood light brightness, in mA, 0..1500

# Options: THE_720_P, THE_800_P, THE_400_P, THE_480_P, THE_1080_P, THE_4_K
RGB_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_720_P
COLOR = True  # Use color camera of mono camera
ISP_SCALE = 1  # Image Signal Processing scale factor

# MANUAL SETTINGS
MANUAL_EXPOSURE = False
EXPOSURE = 5000  # 1-33000
ISO = 400  # 100-1600

MANUAL_FOCUS = False
FOCUS = 130  # 0-255

MANUAL_WHITEBALANCE = False
WHITEBALANCE = 5000  # 1000-12000 (blue -> yellow)

# CAMERA VISUALISATION CONFIG
SHOW_VIDEO = True  # Show video output for each camera
SHOW_POINT_CLOUD = False  # Show point cloud window for each camera
POINT_SIZE = 2

# DEPTH CONFIG
LRCHECK = True  # Better handling for occlusions
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
MEDIAN = (
    dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
)  # Reduce noise and smoothen the depth map.
EXTENDED = True  # Closer-in minimum depth (35cm vs 70cm in normal), disparity range is doubled  # noqa: E501
SUBPIXEL = True  # Better accuracy for longer distance, fractional disparity 32-levels
CONFIDENCE_THRESHOLD = 180  # 0-255, 255 = low confidence, 0 = high confidence

# DEPTH POST PROCESSING
SPECKLE = True  # Reduce the speckle noise
TEMPORAL = True  # Improve the depth data persistency
SPATIAL = False  # Fill invalid depth pixels with valid neighboring depth pixels
MIN_RANGE = 100  # mm
MAX_RANGE = 2000  # mm
DECIMATION_FACTOR = 1  # Reduce the depth scene complexity

# POINTCLOUD GENERATION
DOWNSAMPLE = False  # Voxel downsampling
VOXEL_SIZE = 0.0005  # Voxel downsampling size
REMOVE_NOISE = False  # Remove noise from point cloud

CROP = False  # Crop point cloud
CROP_PERCENT = 0.5  # Crop percentage of pointcloud (0.0-1.0)

# ICP CONFIG
VOXEL_RADIUS = [0.01]  # Voxel radius for ICP
ICP_MAX_ITER = [50]  # Maximum number of ICP iterations

# VOXEL_RADIUS = [0.04, 0.02, 0.01]
# ICP_MAX_ITER = [50, 30, 14]


# CALIBRATION
CALIBRATION_DATA_DIR = "calibration_data"  # Path to camera extrinsics

CHECKERBOARD_SIZE = [12, 8]  # number of squares on the checkerboard
SQUARE_LENGTH = 0.0215  # size of a square in meters

CHARUCO_CALIBRATION = True  # Use charuco calibration
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_LENGTH = 0.017  # size of a marker in meters
