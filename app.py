from flask import Flask, request, jsonify
from flask_cors import CORS
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
app = Flask(__name__)
CORS(app)

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
    
    scale_factor = 4
    new_width = int(cropped_image.shape[1] * scale_factor)
    new_height = int(cropped_image.shape[0] * scale_factor)

    interpolation = cv2.INTER_LINEAR

    enlarged_img = cv2.resize(cropped_image, (new_width, new_height), interpolation=interpolation)
    
    cv2.imwrite('./data/result/output-opt.png', enlarged_img)
    img = np.array(enlarged_img)
    
    print(reader.readtext(img, detail = 0))
    print('-----pytesseract---------------------')
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
    word_boxes = []
    number_pattern = re.compile(r'^\d+(\.\d+)?$')
    # n_boxes = sorted(word_boxes, key=lambda x: (x['top'], x['left']))

    for i in range(n_boxes):
        if d['text'][i].strip():  # Check if the text is not empty (to avoid spaces)
            word_info = {
                'text': d['text'][i],
                'left': d['left'][i],
                'top': d['top'][i],
                'width': d['width'][i],
                'height': d['height'][i],
                'confidence': float(d['conf'][i])
            }
            if number_pattern.match(word_info['text']):
                # If the text matches the number pattern, add a 'is_number' field to the dictionary
                word_info['is_number'] = True
            else:
                # Otherwise, add a 'is_number' field and set it to False
                word_info['is_number'] = False
            word_boxes.append(word_info)
            
    for word_info in word_boxes:
        text = word_info['text']
        confidence = str(word_info['confidence'])  
        is_number = str(word_info['is_number'])  
        print(f"Text: {text}, Confidence: {confidence}, Number: {is_number}")
        x, y, w, h = word_info['left'], word_info['top'], word_info['width'], word_info['height']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return scores[0][0]

@app.route('/upload', methods=['POST'])
def upload_file():
    print(pytesseract.get_languages(config=''))
    if 'image' not in request.files:
        return "No photo part in the request"
    
    img = request.files['image']
    if img and (img.filename.endswith('.png') or img.filename.endswith('.jpeg') or img.filename.endswith('.jpg')):
        image_stream = img.read()
        image_array = cv2.imdecode(np.frombuffer(image_stream, np.uint8), cv2.IMREAD_COLOR)
        # Process the image_array as needed
    if img.filename == '':
        return "No selected Image"
    load_model()
    img = np.array(image_array)
    score = detect(img, True)
    print('score: ', score)
    return "Image uploaded and processed successfully!"

@app.route('/', methods=['GET'])
def ocr():    
    return 'File uploaded successfully', 200


if __name__ == '__main__':
    app.run(debug=True)
    
    
    

