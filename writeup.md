**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/not_car.png
[image3]: ./examples/HOG.png
[image4]: ./examples/sliding_windows.jpg
[image5]: ./examples/sliding_window.jpg
[image6]: ./examples/pipeline.jpg
[image7]: ./examples/original_and_heat.jpg
[image8]: ./examples/heatmap_average.jpg
[image9]: ./examples/output.jpg
[video1]: ./project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `get_hog_features` function in `utils.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`:

![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I ended up using the same values as in the lessons: `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`; 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code is in `utils.py` in `train_classifier`.

I trained a linear SVM using the same features as described in the lectures: I applied a color transform to `YCrCb`, binned color features, as well as histograms of color and obiously HOG features for each channel.

I randomized the data and split a training and test sets.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code is in `utils.py` in `find_cars_multiple_scales`.

The code searches at three different scales that I eyeballed. Here is a visualisation of all the scales:

![alt text][image4]

and a single scale that shows all the windows searched at that scale:

![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code is in ``utils.py`` in ``Pipeline#__process_image`` method

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Once I had it for a single frame, I added those to a list of previous bounding boxes (which I keep for last 10 frames), and repeat the heatmap and label process for all those bounding boxes and draw those on the frame. This helped filter out noise and keep the bounding boxes a little more stable. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image8]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image9]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In general the pipeline is pretty robust, but it is very slow to compute (definitely not real-time), therefore it would help to reduce the number of windows further. Maybe there is a way to search at a larger scale and then focus the search on a specific area once enough matches are found.

Better smoothing techniques would help stabilize the windows, e.g. once a car is found, trying to predict it's possible positions in the next frame.
