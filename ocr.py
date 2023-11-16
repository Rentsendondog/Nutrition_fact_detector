import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time
from crop import crop
import easyocr
import pytesseract
import re
from detect_table_class import NutritionTableDetector

from roi_selection import detect_lines
def load_model():
    """
    load trained weights for the model
    """    
    global obj, reader
    obj = NutritionTableDetector()
    
    reader = easyocr.Reader(['en']) 
    print ("Weights Loaded!")
    
def detect(image, debug):
    # Хүснэгтээ ялгах. 
    boxes, scores, classes, num  = obj.get_classification(image)

    width = image.shape[1]
    height = image.shape[0]

    ymin = boxes[0][0][0]*height
    xmin = boxes[0][0][1]*width
    ymax = boxes[0][0][2]*height
    xmax = boxes[0][0][3]*width

    coords = (xmin, ymin, xmax, ymax)
    cropped_image = crop(image, coords, "./data/result/output.jpg", 0, True)
    
    scale_factor = 1.7
    new_width = int(cropped_image.shape[1] * scale_factor)
    new_height = int(cropped_image.shape[0] * scale_factor)
    
    
    interpolation = cv2.INTER_LINEAR

    enlarged_img = cv2.resize(cropped_image, (new_width, new_height), interpolation=interpolation)
    figure = plt.figure(figsize=(10, 18))
    plt.subplot(1, 2, 1)
    detect_lines(enlarged_img, minLinLength=100, display=True, write = True)
    plt.title('Detected Lines')

    cv2.imwrite('./data/result/output-opt.png', enlarged_img)
    
    img = np.array(enlarged_img)
    
    results = reader.readtext(img, detail=1)
    box_heights = []
    for (bbox, text, prob) in results:
        # Extract the coordinates of the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        
        # Convert coordinates to integers
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        font_size = 0.5 
        height = bottom_right[1] - top_left[1]
        box_heights.append(height)
        # Draw the bounding box on the image
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle

        # Add the text label above the bounding box
        cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)  # Red text

    cv2.namedWindow('Image with Bounding Boxes', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image with Bounding Boxes', new_width, new_height)
    cv2.imshow('Image with Bounding Boxes', img)
    x, y, width, height = cv2.getWindowImageRect('Image with Bounding Boxes')
    print(f"Current window size: {width} x {height}")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Text with bounding boxes')    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return scores[0][0]

if __name__ == "__main__":
    load_model()

    image = cv2.imread('nutrition-fact.png')
    debug = True

    score = detect(image, debug)

    print("Detection Score:", score)