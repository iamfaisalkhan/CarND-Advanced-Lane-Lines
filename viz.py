# -*- coding: utf-8 -*-
# @Author: Faisal Khan
# @Date:   2017-02-23 16:58:07
# @Last Modified by:   Faisal Khan
# @Last Modified time: 2017-02-23 17:37:30

import numpy as np
import cv2

def thumbnail(img):

    if img.ndim == 2:
        img = np.uint8(255 * img/np.max(img))
        img = np.dstack((img, img, img))

    return cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)

def image_mosaic(mainImage, *images):
    mosaic = np.zeros((1080, 1920, 3), dtype=np.uint8)

    cords = [
        [0, 240, 1280, 1600],
        [0, 240, 1600, 1920],
        [240, 480, 1280, 1600],
        [240, 480, 1600, 1920],
    ]

    mosaic[0:720, 0:1280] = mainImage
    for ind, image in enumerate(images):
        cord = cords[ind]
        mosaic[cord[0]:cord[1], cord[2]:cord[3]] = thumbnail(image)

    return mosaic

image0 = cv2.imread('test_images//test1.jpg')
image1 = cv2.imread('test_images//test2.jpg')
image2 = cv2.imread('test_images//test3.jpg')

mosaic = image_mosaic(image0, image1, image2)

cv2.imshow('img', mosaic)
cv2.imwrite('test_images/mosaic.jpg', mosaic)

