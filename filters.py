
import numpy as np
import cv2
import pickle

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    orient_mask = [1, 0]
    if orient == 'y':
        orient_mask = [0, 1]
    
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient_mask[0], orient_mask[1])
    # 3) Take the absolute value of the derivative or gradient
    
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel) )
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled)
    # 6) Return this mask as your binary_output image
    binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    
    sobelxy = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    
    scaled_sobelxy = np.uint8(255 * sobelxy / np.max(sobelxy) )
    # 5) Create a binary mask where mag thresholds are met
    mask =  ((scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1]))
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(scaled_sobelxy)
    binary_output[mask] = 1
    
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    gradient_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(abs_sobelx)
    binary_output[(gradient_dir >= thresh[0]) & (gradient_dir <= thresh[1])] = 1
    
    return binary_output

def color_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[ (s_channel > sthresh[0]) & (s_channel <= sthresh[1]) ] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[ (v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1

    return output

def color_filter(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    yellow = cv2.inRange(hls, (10, 0, 200), (40, 200, 255))
    white = cv2.inRange(hls, (10, 200, 150), (40, 255, 255))
    
    output = np.zeros_like(hls[:, :, 1])

    nonzero = yellow.nonzero()
    output[nonzero[0], nonzero[1]] = 1
    nonzero = white.nonzero()
    output[nonzero[0], nonzero[1]] = 1

    return output

    
