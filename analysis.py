import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import utils
import numpy as np
from params import * 
import matplotlib.pyplot as plt

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

visualise_hog()