import cv2
import numpy as np

class LaneTracker():
    def __init__(self, window_width=80, window_height=80, margin=10, ym_pix=1, xm_pix=1, smoothing_factor=15):
        # Width and height of the sliding window
        self.window_width = 80
        self.window_height = window_height

        self.margin = margin

        # factor specifying number of centroids to be averaged.
        self.smoothing_factor = smoothing_factor

        self.recent_centers = []

        self.ym_per_pix = ym_pix
        self.xm_per_pix = xm_pix
        

    def sliding_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height

        margin = self.margin

        window_centroids = []
        window = np.ones(window_width)

        l_sum = np.sum(warped[int(3 * warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2
        r_sum = np.sum(warped[int(3 * warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(warped.shape[1]/2)

        window_centroids.append((l_center, r_center))

        for level in range(1, (int)(warped.shape[0]/window_height)):

            # Computer histogram of the window between level 2 and level 1
            # Each level window is equal to the portion of the image from the bottom up. 
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):
                                 int(warped.shape[0]-(level)*window_height), :], axis=0)

            # Computed the convolution of the image histogram with the window. 
            conv_signal = np.convolve(window, image_layer)

            offset = window_width/2
            
            # For left center, find the maximum singla in the convolve signal between left min and max index
            # Left  indices are previous left center +/- a margin
            # Right indices are previous right center +/- a margin
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(max(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(max(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)

        return np.average(self.recent_centers[-self.smoothing_factor:], axis=0)
