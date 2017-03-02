
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

class PerspectiveTransform:
    def __init__(self, debug = False):
        # Copy of the previous image use for 
        self.recent_img = None

        # Debug images are only generated if debug parameter is True. 
        self.threshold_img = None
        self.lanes_img = None
        self.pts_img = None
        self.canny_img = None

        # If mosaic is true, return a image mosaic. 
        self.debug = debug

        # If we preivously computed source points, just use them. 
        self.last_src_points = None

        self.backoff = 35

    def _apply_roi(self, img):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `roi`. The rest of the image is set to black.
        """
        # Region of interest
        imshape = img.shape
        roi = np.array([[
                    (70,imshape[0]),
                    (imshape[1]/2, imshape[0]/2), 
                    (imshape[1]/2+10, imshape[0]/2), 
                    (imshape[1]-70,imshape[0])
                ]], dtype=np.int32)

        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, roi, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def _fit_ransac_model(self, pts):
        '''
        Fit a linear regression using RANSAC and returns the slope/intercept
        of the line.
        '''
        X = pts[:, [0, 2]].reshape(-1, 1)
        y = pts[:, [1, 3]].reshape(-1, 1)

        model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), 
                                                    loss="squared_loss",
                                                    max_trials=150
                                                    )
        model_ransac.fit(X, y)

        return model_ransac.estimator_.coef_[0][0],  model_ransac.estimator_.intercept_[0]

    def _computer_corners(self, rslope, rintercept, lslope, lintercept, ysize=700):

        xi = int( (lintercept - rintercept) / (rslope - lslope))
        yi = (rslope * xi + rintercept)

        ry1 = yi + self.backoff
        rx1 = int ( (ry1 - rintercept) / rslope )
        ry2 = ysize - 30
        rx2 = int ( (ry2 - rintercept) / rslope)

        ly1 = yi + self.backoff
        lx1 = int ( ( ly1  - lintercept) / lslope )
        ly2 = ysize - 30
        lx2 = int ( ( ly2 - lintercept) / lslope )

        return [(lx1, ly1), (rx1, ry1), (rx2, ry2), (lx2, ly2)]

    def _get_threshold_img(self, img):
        ksize = 9 # Choose a larger odd number to smooth gradient measurements

        # # Apply each of the thresholding functions
        mag_binary = mag_thresh(img, ksize, (131, 255))
        dir_binary = dir_threshold(img, ksize, (0.5, np.pi))
        color_binary = color_threshold(img, (170, 255), (175, 255))
        lane_binary = white_yellow_binary(img)

        combined = np.zeros_like(img[:,:,0])
        # combined[lane_binary == 1] = 255
        combined[ ( (mag_binary == 1) & (dir_binary == 1) ) | color_binary == 1] = 255

        return combined

    def _get_hough_lines(self, img):
        canny_low = 50.0
        canny_high = 100.0

        rho = 1
        theta = np.pi/180
        hough_threshold = 30
        min_line_len = 10
        max_line_gap = 5

        imshape = img.shape

        img = self._get_threshold_img(img)
        img = self._apply_roi(img)

        edges = cv2.Canny(img, canny_low, canny_high)

        if (self.debug):
            self.canny_img = np.copy(edges)

        lines = cv2.HoughLinesP(edges, rho, theta, hough_threshold, min_line_len, max_line_gap)

        if self.debug:
            self.lanes_img = np.copy(self.recent_img)
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(self.lanes_img, (x1, y1), (x2, y2), [0, 0, 255], 3)

        return lines

    def _get_lane_corners(self, img):

        lines = self._get_hough_lines(img)
        # Vectorized slope computation from all the detected lines. 
        lines = np.squeeze(lines, axis=(1))
        delta = lines.dot(np.array([0, 1, 1, 0, 0, -1, -1, 0]).reshape(4, 2))
        llen = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
        slope = delta[:, 0] / delta[:, 1]

        left_lines = lines[slope < 0]
        right_lines = lines[slope >= 0]

        lslope, lintercept = self._fit_ransac_model(left_lines)
        rslope, rintercept = self._fit_ransac_model(right_lines)

        pts = self._computer_corners(rslope, rintercept, lslope, lintercept, img.shape[0])

        if self.debug:
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255]]
            for i, pt in enumerate(pts):
                 cv2.circle(self.recent_img, (int(pt[0]), int(pt[1])), 5, colors[i], 5)

        return np.float32([
                [pts[0][0], pts[0][1]],
                [pts[1][0], pts[1][1]],
                [pts[2][0], pts[2][1]],
                [pts[3][0], pts[3][1]]
            ])

    def get_warped_image(self, img, rerun=False):

        if self.debug:
            self.recent_img = np.copy(img)

        src = None

        if rerun == False and self.last_src_points != None:
            src = self.last_src_points
        else:
            try:
                src = self._get_lane_corners(img)
                self.last_src_points = src
            except Exception as e:
                print ("Automatic perspective point failed")

                bot_width = .76
                mid_width = .08
                height_pct = .62
                bottom_trim = .935
                 # Static source perspective points in case automatic method fails. 
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

        img = self._get_threshold_img(img)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        return M, Minv, warped

    def get_debug_imgs(self):
        return self.recent_img, self.canny_img, self.lanes_img
