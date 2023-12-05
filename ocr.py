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

    reader = easyocr.Reader(["en"])
    print("Weights Loaded!")


def detect(image, debug):
    # Хүснэгтээ ялгах.
    boxes, scores, classes, num = obj.get_classification(image)

    width = image.shape[1]
    height = image.shape[0]

    ymin = boxes[0][0][0] * height
    xmin = boxes[0][0][1] * width
    ymax = boxes[0][0][2] * height
    xmax = boxes[0][0][3] * width

    coords = (xmin, ymin, xmax, ymax)
    cropped_image = crop(image, coords, "./data/result/output.jpg", 0, True)

    scale_factor = 1.6
    new_width = int(cropped_image.shape[1] * scale_factor)
    new_height = int(cropped_image.shape[0] * scale_factor)

    interpolation = cv2.INTER_LINEAR

    enlarged_img = cv2.resize(
        cropped_image, (new_width, new_height), interpolation=interpolation
    )
    # figure = plt.figure(figsize=(10, 14))
    # plt.subplot(1, 2, 1)
    # detect_lines(enlarged_img, minLinLength=100, display=True, write = True)
    # plt.title('Detected Lines')

    # cv2.imwrite('./data/result/output-opt.png', enlarged_img)

    img = np.array(enlarged_img)
    # ocr хийсэн текстүүдийг хадгалах

    results = reader.readtext(img, detail=1)
    print("-----pytesseract---------------------")
    # мөр бүрийн хувьд текстүүдийг нэгтгэн хадгалах dictionary
    rows = {}
    # avg_line  - г хадгалах dictionary
    rows_sup = {}

    # дундаж шугам нь хайрцагыг огтлолцох эсэхийг шалгах

    def inrow(avg_line, bbox):
        if (avg_line >= min(bbox[0][1], bbox[1][1])) and (
            avg_line <= max(bbox[2][1], bbox[3][1])
        ):
            return True
        return False

    # Loop over all OCR results
    for bbox, text, prob in results:
        avg_line = int((bbox[0][1] + bbox[2][1]) / 2)

        if (
            (avg_line not in rows)
            and (avg_line not in rows_sup)
            or (not inrow(avg_line, bbox))
        ):
            rows[avg_line] = {"text": text, "bbox": bbox}
            rows_sup[avg_line] = avg_line
        else:
            rows[avg_line]["text"] += " " + text
            rows_sup[avg_line] = avg_line

            rows[avg_line]["bbox"] = (
                min(rows[avg_line]["bbox"][0], bbox[0]),
                min(rows[avg_line]["bbox"][1], bbox[1]),
                max(rows[avg_line]["bbox"][2], bbox[2]),
                max(rows[avg_line]["bbox"][3], bbox[3]),
            )
        # draw bounding box
        cv2.rectangle(
            img,
            (int(rows[avg_line]["bbox"][0][0]), int(rows[avg_line]["bbox"][0][1])),
            (int(rows[avg_line]["bbox"][2][0]), int(rows[avg_line]["bbox"][2][1])),
            (0, 255, 0),
            2,
        )

    # Convert the dictionary of rows to a list
    rows_list = list(rows.values())
    # sort y, x
    rows_list.sort(key=lambda r: (r["bbox"][0][1], r["bbox"][0][0]), reverse=False)

    # Print the OCR results
    for row in rows_list:
        print(f"Row Text: {row['text']}")
        print(f"Row Bounding Box: {row['bbox']}")
        print()

        # Add the text label above the bounding box
        # cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)  # Red text

    # Гарж ирэх ёстой текстүүд
    item_list = [text for (bbox, text, prob) in results]
    print(item_list)

    # Дундаж шугам нь хайрцаг бүрийн
    for avg_line in rows_sup.values():
        row = rows[avg_line]
        cv2.line(
            img,
            (int(row["bbox"][0][0]), int(avg_line)),
            (int(new_width - 10), int(avg_line)),
            (255, 0, 255),
            2,
        )

    # дундаж шугам бүрийн текстүүдийг нэгтгэх
    info = []
    for avg_line in rows_sup.values():
        dict = {"txt": ""}
        _r = {}
        for bbox, text, prob in results:
            if inrow(avg_line, bbox):
                if _r is None:
                    dict["txt"] = text
                    _r[avg_line] = avg_line
                else:
                    dict["txt"] += "_"
                    dict["txt"] += text
                    _r[avg_line] = avg_line
        info.append(dict)

    print("---info---\n\n")
    print(info)

    cv2.namedWindow("Image with Bounding Boxes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image with Bounding Boxes", new_width, new_height)
    cv2.imshow("Image with Bounding Boxes", img)
    # x, y, width, height = cv2.getWindowImageRect('Image with Bounding Boxes')
    print(f"Current window size: {new_width} x {new_height}")
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Text with bounding boxes')
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return scores[0][0]


if __name__ == "__main__":
    load_model()

    image = cv2.imread("./nutrition-fact.png")
    debug = True
    score = detect(image, debug)

    print("Detection Score:", score)
