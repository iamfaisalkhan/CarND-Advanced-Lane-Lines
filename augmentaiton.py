import cv2
import numpy as np
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_horizon_y(img, draw=False, min_y=200, max_y=300, hough_threshold=150):
  ''' Estimate horizon y coordinate using Canny edge detector and Hough transform. '''

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray,20,150,apertureSize = 3)

  lines = None
  horizon = None
  horizon_y = 1000

  while lines is None or horizon is None:

    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_threshold)

    if lines is None:
      hough_threshold = hough_threshold - 10
      continue

    horizontal_lines = []

    for i, line in enumerate(lines):
      for rho,theta in line:

        # just the horizontal lines
        if np.sin(theta) > 0.9999:

          if rho < horizon_y and rho >= min_y and rho <= max_y:
            horizon_y = rho
            horizon = line

    if horizon is None:
      hough_threshold = hough_threshold - 10

  if draw and not horizon is None:

    for rho,theta in horizon:
      a = np.cos(theta)
      b = np.sin(theta)

      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1280*(-b))
      y1 = int(y0 + 1280*(a))
      x2 = int(x0 - 1280*(-b))
      y2 = int(y0 - 1280*(a))

      cv2.line(gray,(x1,y1),(x2,y2),(255,255,255),2)

  if horizon is None:
    print('Horizon not found. Return default estimate.')
    return min_y

  return int(horizon_y), gray


images = glob.glob("test_images/test*.jpg")

for idx, fname in enumerate(images):
  img = cv2.imread(fname)
  print (img.shape)
  horizon_y, gray = get_horizon_y(img, draw=True, max_y=img.shape[1]/2)

  print (horizon_y)

  cv2.imshow('img', gray)
  cv2.waitKey(500)



