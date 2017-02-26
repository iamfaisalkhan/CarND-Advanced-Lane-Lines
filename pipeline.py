# -*- coding: utf-8 -*-
# @Author: Faisal Khan
# @Date:   2017-02-22 14:48:00
# @Last Modified by:   Faisal Khan
# @Last Modified time: 2017-02-24 17:26:56

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

from viz import image_mosaic

from tracker import LaneTracker

import matplotlib.pyplot as plt

def getWarpedImage(img):
    bot_width = .76
    mid_width = .08
    height_pct = .62
    bottom_trim = .935

    src = np.float32([
            [img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct],
            [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct],
            [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim],
            [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim]
        ]);
    offset = img.shape[1] * .25

    dst = np.float32([
            [offset, 0],
            [img.shape[1]-offset, 0],
            [img.shape[1]-offset, img.shape[0]],
            [offset, img.shape[0]]
        ]);

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    return M, Minv, warped

def applyFilters(img):
    ksize = 9 # Choose a larger odd number to smooth gradient measurements

    # # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 125))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 125))
        
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(100, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi/2))
    color_binary = color_filter(img, sthresh=(100, 255), vthresh=(50, 255))

    combined = np.zeros_like(img[:,:,0])
    combined [ ( (gradx == 1) & (grady == 1) | color_binary == 1)] = 255

    return combined

def window_mask(window_width, window_height, img, centeroid, level):
    output = np.zeros_like(img)
    output[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),
           max(0, int(centeroid-window_width)):min(int(centeroid + window_width),
            img.shape[1])] =1
    return output

def pipeline(img, mtx, dist, display=False, write=True, write_name='out.jpg'):
    # read in image
    img = cv2.imread(fname)

    # apply camera distortion
    img = cv2.undistort(img, mtx, dist, None, mtx)

    M, Minv, warped = getWarpedImage(img)
    
    binary_warped = applyFilters(warped)

    window_width = 25
    window_height = 80

    tracker = LaneTracker(window_width, window_height, 10, 30/720, 3.7/700)

    window_centroids = tracker.sliding_window_centroids(binary_warped)

    l_points = np.zeros_like(binary_warped)
    r_points = np.zeros_like(binary_warped)

    rightx = []
    leftx = []

    for level in range(0, len(window_centroids)):

        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][1], level)

        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    template = np.array(cv2.merge((l_points, r_points, np.zeros_like(l_points))), np.uint8)
    warpage = np.array(cv2.merge( (binary_warped, binary_warped, binary_warped)), np.uint8)
    warpage = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    yvals = np.arange(0, binary_warped.shape[0])

    # y value of the window centroid
    y_centers = np.arange(binary_warped.shape[0]-(window_height/2), 0, -window_height)

    # Compute polynomial fit
    left_fit = np.polyfit(y_centers, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(y_centers, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(
                    np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0),
                    np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(
                    np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0),
                    np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(zip(
                    np.concatenate((left_fitx+window_width/2, right_fitx[::-1]+window_width/2), axis=0),
                    np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)

    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 0.4, 0.0)

    ym_per_pix = tracker.ym_per_pix
    xm_per_pix = tracker.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(y_centers, np.float32)*ym_per_pix, np.array(leftx, np.float32) * xm_per_pix, 2)
    curverad = (( 1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])


    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - binary_warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, 
            'Radius of Curvature = ' + str(round(curverad)) + '(m)', 
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255),
            2
    )
    cv2.putText(result,
            'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center',
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
    )

    diagnosticImg = image_mosaic(result, img, warped, warpage)

    if write:
        cv2.imwrite(write_name, diagnosticImg)

    if display:
        cv2.imshow('img', diagnosticImg)
        cv2.waitKey(500)

if __name__ == "__main__":
    # Load previously calibration camera calibraton parameters.
    # If camera is not calibrated, look at the calibration.py for howto do it. 
    mtx, dist = load_calibration_matrix('camera_cal/dist_pickle.p')
    images = glob.glob("test_images/test*.jpg")

    for idx, fname in enumerate(images):
        write_name = "./test_images/tracked" + str(idx) + ".jpg"
        pipeline(fname, mtx, dist, display=False, write=True, write_name=write_name)

        

