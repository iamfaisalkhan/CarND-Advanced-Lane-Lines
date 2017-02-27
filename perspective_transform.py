
import cv2
import numpy as np
import glob

from viz import image_mosaic
from filters import color_filter

def filter_yellow_white(image):
    """ 
    A simple filter that detects the presence of white and yellow lanes
    in the image. 
    
    It first converts the image in HSV colorspace ( A better option might be HSL), 
    the masks color for yello and blue. 
    
    `image` : The 3-channel RGB image. 
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Yello color range
    lower_yellow = np.array([20, 100, 200])
    upper_yellow = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(image, image, mask=mask)

    # White color range. 
    sensitivy = 45
    lower_white = np.array([0, 0, 255 - sensitivy])
    upper_white = np.array([255, sensitivy, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res2 = cv2.bitwise_and(image, image, mask=mask)
    
    # Finally combine the white and yello image to get a single image. 
    return cv2.bitwise_or(res, res2)

def filter_yellow_white2(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    yellow = cv2.inRange(hls, (10, 0, 200), (40, 200, 255))
    res = cv2.bitwise_and(img, img, mask=yellow)

    white = cv2.inRange(hls, (10, 200, 150), (40, 255, 255))
    res2 = cv2.bitwise_and(img, img, mask=white)

    return cv2.bitwise_or(res, res2)


def region_of_interest(img, vertices):
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

def compute_image_plane_pts(img):
    canny_low = 50.0
    canny_high = 100.0

    rho = 1
    theta = np.pi/180
    hough_threshold = 30
    min_line_len = 10
    max_line_gap = 5

    src_img = np.copy(img)
    result = np.copy(img)

    img = filter_yellow_white2(img)
    imshape = img.shape
    vertices = np.array([[
                (70,imshape[0]),
                (imshape[1]/2, imshape[0]/2), 
                (imshape[1]/2+10, imshape[0]/2), 
                (imshape[1]-70,imshape[0])
            ]], 
                        dtype=np.int32)

    img = region_of_interest(img, vertices)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    
    lines = cv2.HoughLinesP(edges, rho, theta, hough_threshold, min_line_len, max_line_gap)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            angle = np.arctan2((y2-y1), (x2-y1)) * 180.0/np.pi
            cv2.line(result, (x1, y1), (x2, y2), color=[255, 0, 0], thickness=5)
        
    result = image_mosaic(src_img, img, result)

    while (1):
        cv2.imshow('img', result)

        k = cv2.waitKey(100)
        k -= 0x100000
        if k == 27:
            break

if __name__ == "__main__":
    images = glob.glob("test_images/test*.jpg")

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        compute_image_plane_pts(img)
        # if idx == 2:
            # break
        

