import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import utils
from params import *  
import os
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

vehicles_glob = 'vehicles/**/*.png'
non_vehicles_glob = "non-vehicles/**/*.png"

# Smoothing
n_frames = 10
single_frame_threshold = 4
average_threshold = 6

def load_images_from_directory(glob_pattern):
    images = []

    for image in glob.glob(glob_pattern):
        images.append(image)

    return images

def train_classifier():
    cars = load_images_from_directory(vehicles_glob)
    notcars = load_images_from_directory(non_vehicles_glob)

    (svc, X_scaler) = utils.train_classifier(cars,
                                  notcars,
                                  color_space, 
                                  spatial_size, 
                                  hist_bins, 
                                  hist_bins_range, 
                                  orient, 
                                  pix_per_cell, 
                                  cell_per_block, 
                                  hog_channel, 
                                  spatial_feat, 
                                  hist_feat, 
                                  hog_feat)

    joblib.dump(X_scaler, X_scaler_file) 
    joblib.dump(svc, model_file) 

    return (svc, X_scaler)

def main():

    svc = None
    X_scaler = None
    if not os.path.isfile(model_file) or not os.path.isfile(X_scaler_file):
        print("Model not found, training")
        (svc, X_scaler) = train_classifier()
    else:
        print("Found model, loading")
        svc = joblib.load(model_file) 
        X_scaler = joblib.load(X_scaler_file)
 
    pipeline = utils.Pipeline('test_video',
                              scales,
                              color_space,
                              ystart, 
                              ystop, 
                              svc, 
                              X_scaler, 
                              orient, 
                              pix_per_cell, 
                              cell_per_block, 
                              spatial_size, 
                              hist_bins,
                              hist_bins_range,
                              n_frames,
                              single_frame_threshold,
                              average_threshold)
    pipeline.process_video()

if __name__ == "__main__":
    main()