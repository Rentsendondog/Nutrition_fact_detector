import sys
import math
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

           
def detect_lines(image, title='default', rho = 1, theta = 2, threshold = 50, minLinLength = 290, maxLineGap = 6, display = False, write = False):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    if gray is None:
        print ('Error opening image!')
        return -1
    
    dst = cv.Canny(gray, 50, 150, None, 3)
    
    # plt.subplot(121), plt.imshow(gray, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    # plt.subplot(122), plt.imshow(dst, cmap='gray')
    # plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

    # plt.show()
    lines = cv.HoughLinesP(dst, 1, np.pi/180, threshold=80, minLineLength=80, maxLineGap=10)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv.imshow('Detected Lines', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return 0

