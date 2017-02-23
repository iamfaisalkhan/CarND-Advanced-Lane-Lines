# -*- coding: utf-8 -*-
# @Author: Faisal Khan
# @Date:   2017-02-22 14:48:00
# @Last Modified by:   Faisal Khan
# @Last Modified time: 2017-02-22 16:22:25

import cv2
import numpy as np
import glob
import pickle

from calibration import load_calibration_matrix
from calibration import test_calibration

from filters import abs_sobel_thresh
from filters import mag_thresh
from filters import dir_threshold
from filters import color_filter

import matplotlib.pyplot as plt


def pipeline(img, mtx, dist, display=False, write=True, write_name='out.jpg'):
    # read in image
    img = cv2.imread(fname)

    # apply camera distortion
    img = cv2.undistort(img, mtx, dist, None, mtx)
    
    ksize = 9 # Choose a larger odd number to smooth gradient measurements

    # # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 90))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 90))
        
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(100, 255))
    # dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi/2))
    color_binary = color_filter(img, sthresh=(100, 255), vthresh=(50, 255))

    combined = np.zeros_like(img[:,:,0])
    combined [ ( (gradx == 1) | (grady == 1)) & color_binary == 1] = 255

    result = combined

    if write:
        cv2.imwrite(write_name, result)

    if display:
        cv2.imshow('img', result)
        cv2.waitKey(500)

if __name__ == "__main__":

    # Load previously calibration camera calibraton parameters.
    # If camera is not calibrated, look at the calibration.py for howto do it. 
    mtx, dist = load_calibration_matrix('camera_cal/dist_pickle.p')
    images = glob.glob("test_images/test*.jpg")

    for idx, fname in enumerate(images):
        write_name = "./test_images/tracked" + str(idx) + ".jpg"
        pipeline(fname, mtx, dist, display=False, write=True, write_name=write_name)

        
