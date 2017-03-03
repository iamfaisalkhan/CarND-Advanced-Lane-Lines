#Advanced Lane Lines


[//]: # (Image References)

[image1]: ./imgs/calibration_example.jpg "Calibration Example"
[image2]: ./imgs/test1.jpg "Reference Image"
[image3]: ./imgs/undistorted.jpg "Undistorted Road Image"
[image4]: ./imgs/color_transformation1.jpg "Color/Gradient Transofmraiton without ROI"
[image5]: ./imgs/color_transformation2.jpg "Color/Gradient Transofmraiton with ROI"
[image6]: ./imgs/hough_lines.jpg "Hough Lines"
[image7]: ./imgs/ransac_example.jpg "RANSAC Regression"
[image8]: ./imgs/perspective_src_points.jpg "Perspective source points"
[image9]: ./imgs/perspective_transform.jpg "Perspective transformation"
[image10]: ./imgs/centroids.jpg "Centroids"
[image11]: ./imgs/final.jpg "Final Result"


## Code Structure

A briref overview of what's in the repository.

* **[main.py](./main.py)**: Launches the pipeline. 
* **[pipeline.py](./pipeline.py)** : It contains LanePipeline class that tie together all the components of our pipeline. The process() method process a single image and pipeline state is maintained through class variables.
* **[perspective_transform.py](perspective_transform.py)**: Have the code for PerspectiveTransform class. This class computes the warped image by first detecting 4 lane markers as source points and then reusing them for each subsequent call to get the warped image.
* **[tracker.py](./tracker.py)**: The LaneTracker class have the main logic for sliding window. 
* **[filters.py](./filters.py)**: Have all the color and gradient image filter. 
* **[viz.py](./viz.py)**: Generate debugging image. 

To **launch** the pipeline:

```
python main.py <input-video.mp4> <output-video.mp4> 
```

## Project Component

### Camera Calibration

The calibration code is given in calibration.py:(lines 41-81). 

The calibration was done using OpenCVs `cv2.calibrateCamera()` method. It computes camera's distoration coefficients and camera matrix (contains focal lengths and optical centers) along with translation and rotation vectors [1]. 

The two main arguments to this methods are:

* object points : A set of 3D points of an object in real world space.
* image points : Corresponding 2D points on the same object in image space. 

We use a series of points from images of chessboard patterns taken at different camera positions as object points. The chessboard is assumed to be at fixed X, Y positions with Z=0. The corresponding points in the image space are computed using `cv2.findChessboardCorners()` that returns the corners of the chessboard. 

The undistored image is produced using the `cv2.undistort()`. The calibration parameters are stored as a pickle file for future use. 

![alt text][image1]

###Pipeline 

In this section, we walk you through the pipeline steps using the following image as an example.

![alt text][image2]

#### 1. Undistort Image

The first step in our pipeline is to undistort the image using the calibration matrix computed in the earlier steps. The result is shown below:

![alt text][image3]

####2. Color / Gradient Transformation

We use a combination of color and gradient thresholding to obtain a binary image suitable for finding the points on the lane before fitting a second-degree polynomial. Specifically, the magnitude and direction of image gradient is thresholded along with the filtering of the image based on the white and yellow colors using HSV and HLS color space. More details is in `perspective_transform.py:_get_threshold_img(lines 105-117)`. The `filters.py` has a collection of different image filters used in this project. 

As we are assuming that camera is mounted at the center of the car, we also set a region of interest to seperate area around the lanes from the rest of the image. 

```
imshape = img.shape
roi = np.array([[
                 (70,imshape[0]),
                 (imshape[1]/2, imshape[0]/2), 
                 (imshape[1]/2+10, imshape[0]/2), 
                 (imshape[1]-70,imshape[0])
                ]], dtype=np.int32)
```

The result from this step are shown below (with and without region of interest) 

**Threshold Image Without ROI**

![alt text][image4]

**Threshold Image With ROI**

![alt text][image5]

####3.Perspective Transformation

The code for perspective transform is in `perspective_transform.py:get_warped_image (line 208)`. 

The perspective points are computed automatically by fitting a linear regression line through the left and right lanes and using the slope/intercept of these lines to compute the corners of the lane lines. The hough transformation is used to compute the  candidate lines/points in the image. These points are further divided into the left and right lane points based on the slope value. 

![alt text][image6]

A RANSAC based linear fitting model [2] is used to compute the slope/intercept of the best line passing through these points for both the left and right lanes. 

![alt text][image7]

The four points computed at the end of this process are visualized below. These corner points are used as the source points for our perspective transformation. 

![alt text][image8]

This process is done only once for the first frame and the source points are reused through the tracking session. However as a fall back, if the detection of these 'corner' points fail, we use static points (see `perspective_transform.py:215-235`). The corresponding destination points are set as follows:

```
offset = img.shape[1] * .25
dst = np.float32([
                [offset, 0],
                [img.shape[1]-offset, 0],
                [img.shape[1]-offset, img.shape[0]],
                [offset, img.shape[0]]
     ]);
```             

The final bird eye view of the image using both the original image and threshold image is shown below. 

![alt text][image9]

####4. Cureve Fitting

We use a second-degree polynomial to model the shape of the road. Two separate polynomials are used to model the right and left lanes. A sliding window algorithm is used to compute points on the lanes that are then used to fit these polynomials. The sliding window algorithm first computes candidate positions on the image by computing a column wise histogram of the lower 3/4th of the image. The histogram of the columns is convolved with a 1-D signal to find the peaks corresponding to two lanes in the warped image. After finding two candidates points, a window is slided through the height of the image searching for next histogram peaks in the vicinity of these candidate points. 

These steps are implemented in `tracker.py:sliding_window_centroids(line 20-)`. 

The results of the sliding window algorithm is shown below.

![alt text][image10]

####5. Radius of the Curvature.

The radius of the curvature of the road is computed using the formula given here [4]. The implementation is given in `pipeline.py:process()-line#107-117`. 

####6. Final Result.

The pipeline 

![alt text][image11]

---

###Pipeline Videos


1. Result of project video : [project_video.mp4](./project_video.mp4)
2. Challenge video : [challenge_video.mp4](./challenge_video.mp4)

---

###Discussion

Using a combination of color and gradient thresholding we were able to handle, fairly decently, both the project video and the challenge video. However, the harder challenge seems to be more tricky for our pipeline. One way, we could improve things further is to not run the sliding window in each frame and instead used previously detected curve when we can not find the points on the lanes.

The automatic detection of the perspective points makes our pipeline little more generic as we can get a good warped image for different type of videos and possibly camera positions. 

The pipeline generally fails under tricky lighting condition or sharp curves as we are not able to either distinguish between background and lane colors or our window width +/- margin misses the actual lanes. To handle such situations, we need extra constrains and make our sliding window more adaptive based on the sharpness of the curve detected in previous frame(s). 

The RANSAC method use for finding a good linear fit for perspective transform might also be able to handle outliers when fitting a polynomial. This might also improve the overall stability of the tracking. 


#### References
* [1] http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
* [2] http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py
* [3] https://www.youtube.com/watch?v=vWY8YUayf9Q
* [4] http://www.intmath.com/applications-differentiation/8-radius-curvature.php
