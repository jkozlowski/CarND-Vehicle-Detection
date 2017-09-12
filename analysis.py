import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import utils
import numpy as np
from params import * 
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def visualise_hog():
    images = ['examples/car.png', 'examples/not_car.png']
    channels = 3
    fig, axarr = plt.subplots(channels, len(images))
    fig.suptitle('Car vs non car HOG')

    for img_id, img in enumerate(images):
        image = mpimg.imread(img)
        converted = utils.convert_color(image, 'YCrCb')
        converted = converted.astype(np.float32)/255
        for channel in range(channels):
            fv, hog_channel = utils.get_hog_features(converted[:,:,channel], 
                            orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=False)
            axarr[channel, img_id].imshow(hog_channel, cmap='gray')
            axarr[channel, img_id].set_title('Channel({})'.format(channel))
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig('examples/HOG.png')

def visualise_windows():
    svc = joblib.load(model_file) 
    X_scaler = joblib.load(X_scaler_file)
    image = mpimg.imread('test_images/test1.jpg')

    bboxes, boxes = utils.find_cars_multiple_scales(image,
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
        hist_bins_range)

    single_window(np.copy(image), boxes[0])
    all_windows(np.copy(image), boxes)

def single_window(img, single_scale):
    c1 =  np.random.randint(0, 255)
    c2 =  np.random.randint(0, 255)
    c3 =  np.random.randint(0, 255)      
    utils.draw_bboxes(img, single_scale, color=(c1, c2, c3))
    utils.draw_bboxes(img, [single_scale[int(len(single_scale)/2)]])
    
    mpimg.imsave('examples/sliding_window.jpg', img)

def all_windows(img, boxes):
    for box in boxes:
        c1 =  np.random.randint(0, 255)
        c2 =  np.random.randint(0, 255)
        c3 =  np.random.randint(0, 255)      
        utils.draw_bboxes(img, [box[0]], color=(c1, c2, c3))
    
    mpimg.imsave('examples/sliding_windows.jpg', img)

def visualise_pipeline():
    svc = joblib.load(model_file) 
    X_scaler = joblib.load(X_scaler_file)
    image = mpimg.imread('test_images/test1.jpg')

    bboxes, boxes = utils.find_cars_multiple_scales(image,
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
        hist_bins_range)

    utils.draw_bboxes(image, bboxes)
    mpimg.imsave('examples/pipeline.jpg', image)

def visualise_heat():

    start_frame = 6
    num_frames = 6

    fig, axarr = plt.subplots(num_frames, 2, figsize=(20,20))
    fig.suptitle('Frames and heat')

    for id, frame in enumerate(range(start_frame, start_frame + num_frames)):
        original = mpimg.imread('output_images/original-{}.jpg'.format(frame))
        heatmap_single = mpimg.imread('output_images/heatmap-single-{}.jpg'.format(frame))

        axarr[id, 0].imshow(original)
        axarr[id, 0].set_title('Frame({})'.format(frame))
        axarr[id, 1].imshow(heatmap_single, cmap='gray')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig('examples/original_and_heat.jpg')

visualise_hog()
visualise_windows()
visualise_pipeline()
visualise_heat()