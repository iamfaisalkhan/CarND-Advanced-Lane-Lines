# -*- coding: utf-8 -*-
# @Author: Faisal Khan
# @Date:   2017-02-22 14:34:23
# @Last Modified by:   Faisal Khan
# @Last Modified time: 2017-02-22 15:06:02


import cv2
import numpy as np
import glob
import pickle

def load_calibration_matrix(calib_mat_file):
    calibration_mat = None
    with open(calib_mat_file, "rb") as f:
        calibration_mat = pickle.load(f)

    mtx = calibration_mat["mtx"]
    dist = calibration_mat["dist"]

    return mtx, dist

def test_calibration(calib_matfile, test_image):
    import matplotlib.pyplot as plt

    mtx, dist = load_calibration_matrix(calib_matfile)

    img = cv2.imread(test_image)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Imae', fontsize=30)

    plt.show()


def calibrate(images, test_image):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob("%s/calibration*.jpg"%calib_image_dir)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    img = cv2.imread('camera_cal/calibration1.jpg')
    size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist

    pickle.dump(dist_pickle, open("%s/dist_pickle.p"%(calib_image_dir), "wb"))
        
if __name__ == "__main__":
    calibrate("./camera_cal", "./camera_cal/calibration1.jpg", False)