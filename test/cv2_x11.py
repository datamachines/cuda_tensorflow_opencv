# Adapted from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

import numpy as np
import cv2

img = cv2.imread('test-pattern-152459_640.png',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
