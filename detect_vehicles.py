import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import utils
import os
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

vehicles_glob = 'vehicles/**/*.png'
non_vehicles_glob = "non-vehicles/**/*.png"
model_file = 'model.pkl'
X_scaler_file = 'scaler.pkl'

# TODO play with these values to see how your classifier
# performs under different binning scenarios
### TODO: Tweak these parameters and see how the results change.
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
color_space = 'YCrCb'
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16 # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
ystart = 400
ystop = 656
scale = 1.5

orient=9
pix_per_cell=8
cell_per_block=2
spatial_size=(32, 32)
hist_bins=32
hist_bins_range=(0,1)

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

    image = mpimg.imread('test_images/test1.jpg')

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255

    window_img = utils.find_cars(image, 
              color_space,
              ystart, 
              ystop, 
              scale, 
              svc, 
              X_scaler, 
              orient, 
              pix_per_cell, 
              cell_per_block, 
              spatial_size, 
              hist_bins,
              hist_bins_range)

    mpimg.imsave("out.png", window_img)

if __name__ == "__main__":
    main()