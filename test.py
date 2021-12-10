import numpy as np
import cv2 as cv

img = cv.imread('1.jpg')
print(img)
cv.namedWindow('123')
cv.imshow('123',img)
cv.waitKey(0)
cv.destroyAllWindows()