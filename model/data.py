import numpy as np
from ultralytics import YOLO
import cv2
import keras_ocr
import easyocr
import matplotlib.pyplot as plt
import random
import os



model = YOLO("plate_detection_try_2\\weights\\best.pt")
char_model = YOLO("charecter_detection\\try2\\weights\\best.pt")


dir = "all_data\\New Data 2"
file_list = os.listdir(dir)

for i in file_list:
    img_path = os.path.join(dir, i)
    # print(img_path) 
    detections = model(img_path)[0]

    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        print(class_id)
        detections_.append([x1, y1, x2, y2, score])
    
    try:
        fls = detections_[0]
        print(fls)
        frame = cv2.imread(img_path)
        crop = frame[int(fls[1]):int(fls[3]), int(fls[0]):int(fls[2])]
        crop_gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # crop_gray =cv2.bitwise_not(crop_gray)
        ret, crop_gray_thresh = cv2.threshold(crop_gray, 127, 255, cv2.THRESH_BINARY_INV)


        ksize = (1, 1)
    
    # Using cv2.blur() method 
        image = cv2.blur(crop_gray, ksize) 


        # reader = easyocr.Reader(['ch_sim','en'])
        plt.imshow(crop_gray_thresh, cmap="gray")
        # result = reader.readtext(image)
        # print(result)
    except IndexError:
        print("Unable to Detect")

    cv2.imwrite("image.jpg", crop_gray)

    char_detection = char_model("image.jpg")[0]
    char_detections_ = []
    for detection in char_detection.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        char_detections_.append([x1, y1, x2, y2, score])
        # print(class_id)
    
    new_char_list = list(char_detections_)

    frame = cv2.imread('image.jpg')
    print(f"thres {0.1 * ((frame.shape)[1])}")
    for i in char_detections_:    
        fls = i
        

        if (int(fls[3]) - int(fls[1])) < (0.1 * ((frame.shape)[1])) :
            pass
        else:
            # cv2.rectangle(frame, (int(fls[0]), int(fls[1])), (int(fls[2]), int(fls[3])),(255,0,0), 1)
            print(int(fls[3]) - int(fls[1]))
            crop = frame[int(fls[1]):int(fls[3]), int(fls[0]):int(fls[2])]
            crop_gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            proper_image = cv2.resize(crop_gray, (40, 40))
            cv2.imwrite(f"charecters/image{random.random()}.jpg", proper_image)
            

        # adding a little blur 
        ksize = (1, 1)
        image = cv2.blur(crop_gray, ksize) 