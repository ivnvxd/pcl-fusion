# import numpy as np
import open3d
import glob
import os

from helpers import remove_color_points
import config


directory = os.path.expanduser("~/view")
input_folder = os.path.join(directory, "records", "input_data")
output_folder = os.path.join(directory, "records", "output_data")

os.makedirs(output_folder, exist_ok=True)

files = glob.glob("*.ply", root_dir=input_folder)
counter = 1

for file in files:
    print(f"Processing file {counter}/{len(files)}:", file, end="\r")
    counter += 1

    pcd = open3d.io.read_point_cloud(os.path.join(input_folder, file), remove_nan_points=True)
    # pcd = remove_color_points(pcd, config.color_to_remove, config.color_treshold)
    pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1)[0]

    open3d.io.write_point_cloud(os.path.join(output_folder, file), pcd)