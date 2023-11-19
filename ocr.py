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
    
    scale_factor = 1.6
    new_width = int(cropped_image.shape[1] * scale_factor)
    new_height = int(cropped_image.shape[0] * scale_factor)
    
    
    interpolation = cv2.INTER_LINEAR

    enlarged_img = cv2.resize(cropped_image, (new_width, new_height), interpolation=interpolation)
    # figure = plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    # detect_lines(enlarged_img, minLinLength=100, display=True, write = True)
    plt.title('Detected Lines')

    cv2.imwrite('./data/result/output-opt.png', enlarged_img)
    
    img = np.array(enlarged_img)
    
    results = reader.readtext(img, detail=1)
    # box_heights = []
    # boxes = []
    # info = []
    rows = {}

    for (bbox, text, prob) in results:
    # Use the vertical midpoint of the bounding box as a key for the row
        print('bbox[1]', bbox[1], 'bbox[3]', bbox[3])
        row_key = int((bbox[1][1] + bbox[3][1]) / 2)

    # Create a new row dictionary if it doesn't exist
        if row_key not in rows:
            rows[row_key] = {'text': text, 'bbox': bbox}
        else:
            # Concatenate the text if the row already exists
            rows[row_key]['text'] += ' ' + text
            # Update the bounding box to cover the entire row
            rows[row_key]['bbox'] = (
                min(rows[row_key]['bbox'][0], bbox[0]),
                min(rows[row_key]['bbox'][1], bbox[1]),
                max(rows[row_key]['bbox'][2], bbox[2]),
                max(rows[row_key]['bbox'][3], bbox[3])
            )

    # Convert the dictionary of rows to a list
    rows_list = list(rows.values())
    # top_left, bottom_right sorted
    rows_list.sort(key=lambda r: (r['bbox'][0][0], r['bbox'][0][1], r['bbox'][2][1]), reverse=False)

    # Display the results
    for row in rows_list:
        print(f"Row Text: {row['text']}")
        print(f"Row Bounding Box: {row['bbox']}")
        print()

    for elm in rows_list:
        avg_line = (elm['bbox'][1][1] + elm['bbox'][3][1]) / 2
        if avg_line not in rows:
            rows[avg_line] = {'text': elm['text'], 'bbox': elm['bbox']}

        
        # Add the text label above the bounding box
        # cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)  # Red text

    # avg red line
    # for box in boxes:
    #     (top_left, bottom_right) = box
    #     y = bottom_right - top_left / 2
    #     x1 = 0
    #     x2 = img.shape[1] - 10
    #     # draw line
    #     cv2.line(img, (x1, y), (x2, y), (255, 0, 0), 1)

    key_dict = ['text1', 'text2', 'text3', 'text4']

    item_list = [text for (bbox, text, prob) in results]
    print(item_list)




    cv2.namedWindow('Image with Bounding Boxes', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image with Bounding Boxes', new_width, new_height)
    cv2.imshow('Image with Bounding Boxes', img)
    x, y, width, height = cv2.getWindowImageRect('Image with Bounding Boxes')
    print(f"Current window size: {new_width} x {new_height}")
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