
import cv2
import numpy as np
import glob

from filters import white_yellow_binary
from filters import abs_sobel_thresh
from filters import color_threshold
from filters import mag_thresh
from filters import dir_threshold

from viz import image_mosaic

from sklearn import linear_model, datasets

from matplotlib import pyplot as plt

def _region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def _fit_ransac_model(pts):
    '''
    Fit a linear regression using RANSAC and returns the slope/intercept
    of the line.
    '''
    X = pts[:, [0, 2]].reshape(-1, 1)
    y = pts[:, [1, 3]].reshape(-1, 1)

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(X, y)
    return model_ransac.estimator_.coef_[0][0],  model_ransac.estimator_.intercept_[0]

def _computer_corners(rslope, rintercept, lslope, lintercept, ysize=700, backoff=35):
    xi = int( (lintercept - rintercept) / (rslope - lslope))
    yi = (rslope * xi + rintercept)

    ry1 = yi + 40
    rx1 = int ( (ry1 - rintercept) / rslope )
    ry2 = ysize - 10
    rx2 = int ( (ry2 - rintercept) / rslope)

    ly1 = yi + 40
    lx1 = int ( ( ly1  - lintercept) / lslope )
    ly2 = ysize - 10
    lx2 = int ( ( ly2 - lintercept) / lslope )

    return [(lx1, ly1), (rx1, ry1), (rx2, ry2), (lx2, ly2)]

def _gradient_color_threshold(img):

    # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    ksize = 9 # Choose a larger odd number to smooth gradient measurements

    # # Apply each of the thresholding functions
    #gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(0, 50))
    #grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(0, 161))
    mag_binary = mag_thresh(img, ksize, (131, 255))
    dir_binary = dir_threshold(img, ksize, (0.5, np.pi))
    color_binary = color_threshold(img, (170, 255), (175, 255))
    lane_binary = white_yellow_binary(img)

    combined = np.zeros_like(img[:,:,0])
    #combined [ ( (gradx == 1) & (grady == 1) | color_binary == 1)] = 255
    # color_binary = [color_binary == 1 & lane_binary == 1]
    combined[ ( (mag_binary == 1) & (dir_binary == 1) ) | color_binary == 1] = 255

    return combined

def lane_corner_markers(img):
    canny_low = 50.0
    canny_high = 100.0

    rho = 1
    theta = np.pi/180
    hough_threshold = 30
    min_line_len = 30
    max_line_gap = 5

    src_img = np.copy(img)
    hough_img = np.copy(img)

    imshape = img.shape
    #TODO Better yet to define the ROI based on the image
    # percentage. 
    vertices = np.array([[
                (70,imshape[0]),
                (imshape[1]/2, imshape[0]/2), 
                (imshape[1]/2+10, imshape[0]/2), 
                (imshape[1]-70,imshape[0])
            ]], 
                        dtype=np.int32)


    gray = _gradient_color_threshold(img)
    img = _region_of_interest(gray, vertices)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, canny_low, canny_high)
    
    lines = cv2.HoughLinesP(edges, rho, theta, hough_threshold, min_line_len, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(hough_img, (x1, y1), (x2, y2), [0, 0, 255], 3)

    lines = np.squeeze(lines, axis=(1))

    # Vectorized slope computation from all the detected lines. 
    delta = lines.dot(np.array([0, 1, 1, 0, 0, -1, -1, 0]).reshape(4, 2))
    llen = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
    slope = delta[:, 0] / delta[:, 1]

    left_lines = lines[slope < 0]
    right_lines = lines[slope >= 0]

    lslope, lintercept = _fit_ransac_model(left_lines)
    rslope, rintercept = _fit_ransac_model(right_lines)

    pts = _computer_corners(rslope, rintercept, lslope, lintercept, img.shape[0])

    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]

    for i, pt in enumerate(pts):
        cv2.circle(src_img, (int(pt[0]), int(pt[1])), 5, colors[i], 5)

    result = image_mosaic(src_img, gray, edges, img, hough_img)

    k = 0
    while (1):
        cv2.imshow('img', result)

        k = cv2.waitKey(100)
        k -= 0x100000
        if k == 27 or k == 113:
            break

    return k

if __name__ == "__main__":
    images = glob.glob("test_images/test*.jpg")

    for idx, fname in enumerate(images):
        print (fname)
        img = cv2.imread(fname)
        k = lane_corner_markers(img)
        if k == 113:
            break
        # if idx == 2:
            # break
        # break
        


