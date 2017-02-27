# filtertest.py

from calibration import *
from filters import *

from viz import image_mosaic

import cv2
import numpy as np

mtx, dist = load_calibration_matrix('camera_cal/dist_pickle.p')
images = glob.glob("test_images/test*.jpg")

cnt = 0

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    ksize = 9

    color_binary = color_thresholding(img)
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 255))


    combined = np.zeros_like(img[:,:,0])
    combined [ ( (gradx == 1) & (grady == 1) | color_binary == 1)] = 255


    result = image_mosaic(img, gradx, grady, color_binary, combined)

    while (1):
        cv2.imshow('img', result)

        k = cv2.waitKey(100)
        k -= 0x100000
        if k == 27:
            break