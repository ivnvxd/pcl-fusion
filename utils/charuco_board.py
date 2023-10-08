import cv2

dictionary = cv2.aruco.DICT_4X4_50
dict_name = "DICT_4X4_50"

# dictionary = cv2.aruco.DICT_ARUCO_MIP_36h12
# dict_name = "DICT_ARUCO_MIP_36h12"

aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
size = [12, 8]  # number of squares in X and Y directions
square_length = 15  # size of a square in meters
marker_length = 12  # size of a marker in meters

board = cv2.aruco.CharucoBoard(
    size=size,
    squareLength=square_length,
    markerLength=marker_length,
    dictionary=aruco_dict,
)

# board.setLegacyPattern(True)
img_board = board.generateImage((3508, 2480), marginSize=200, borderBits=1)

board_name = (
    "charuco_OpenCV-"
    + cv2.__version__
    + "_"
    + str(size[0])
    + "x"
    + str(size[1])
    + "_"
    + dict_name
)

cv2.putText(
    img_board, board_name, (350, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2
)

cv2.imwrite(
    board_name + ".png",
    img_board,
)
