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

    mosaic[0:720, 0:1280] = mainImage
    for ind, image in enumerate(images):
       mosaic[(ind*240):(ind*240)+240, (index*)] = thumbnail(image)

    return mosaic

image0 = cv2.imread('test_images//test1.jpg')
image1 = cv2.imread('test_images//test2.jpg')
image2 = cv2.imread('test_images//test3.jpg')

mosaic = image_mosaic(image0, image1, image2)

cv2.imshow('img', mosaic)
cv2.waitKey(1000)


