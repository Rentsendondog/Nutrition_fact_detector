import sys
import math
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

           
def detect_lines(image, title='default', rho = 1, theta = 2, threshold = 50, minLinLength = 290, maxLineGap = 6, display = False, write = False):
    # grayscale хийх
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    if gray is None:
        print ('Error opening image!')
        return -1
    
    # Ирмэгүүдийг таних, тодоруулах
    dst = cv.Canny(gray, 50, 150, None, 3)

    # grayscale хийсэн зургийг харуулах
    # plt.subplot(121), plt.imshow(gray, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # Canny -аар ирмэгүүдийг тодоруулсан зургийг харуулах
    # plt.subplot(122), plt.imshow(dst, cmap='gray')
    # plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Хүснэгтийн ирмэгүүдийг тодоруулах
    lines = cv.HoughLinesP(dst, 1, np.pi/180, threshold=80, minLineLength=80, maxLineGap=9)
    # sort y1, x1
    lines = sorted(lines, key=lambda x: (x[0][1], x[0][0]))
    print('----len: ',len(lines))
    cur_y = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - cur_y) > 15 and (y1 == y2):
            x2 = image.shape[1]- 10
            x1 = 2
            cur_y = y1
            cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv.imshow('Detected Lines', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    return 0

