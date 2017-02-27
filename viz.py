# -*- coding: utf-8 -*-
# @Author: Faisal Khan
# @Date:   2017-02-23 16:58:07
# @Last Modified by:   Faisal Khan
# @Last Modified time: 2017-02-27 16:42:08

import numpy as np
import cv2

def thumbnail(img, size=(320, 240)):

    if img.ndim == 2:
        img = np.uint8(255.0 * img/np.max(img))
        img = np.dstack((img, img, img))

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def image_mosaic(mainImage, *images):
    mosaic = np.zeros((1080, 1920, 3), dtype=np.uint8)

    placements = [
        [0, 360, 1280, 1920],
        [360, 720, 1280, 1920],
        [720, 1080, 1280, 1920],
        [720, 1080, 640, 1280],
        [720, 1080, 0, 640],
    ]

    if mainImage.shape[0] != 1280 or mainImage.shape[1] != 720:
        mainImage = cv2.resize(mainImage, (1280, 720), interpolation=cv2.INTER_AREA)

    mosaic[0:720, 0:1280] = mainImage

    for ind, image in enumerate(images):
        cord = placements[ind]
        size = (cord[3] - cord[2], cord[1] - cord[0])
        mosaic[cord[0]:cord[1], cord[2]:cord[3]] = thumbnail(image, size)

    return mosaic


