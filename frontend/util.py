from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import easyocr
reader = easyocr.Reader(['ch_sim','en'])

model = YOLO("artifacts\\detection\\best.pt")

def detect_number_plate(img_path):
    frame = cv2.imread(img_path)
    detections = model(img_path)[0]
    detections_ = []
    try:
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            detections_.append([x1, y1, x2, y2, score])
        obj = detections_[0]

        cv2.rectangle(frame, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])),(255,0,0), 10)
        return int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), frame
    except IndexError:
        # print("Unable to Detect")
        cv2.putText(frame, "Not Found !" , (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (15,255,255), 3)
        return 0, 0, 0, 0, frame

def number_plate_crop(img_path):
    x1, x2, y1, y2, frame = detect_number_plate(img_path)

    if x1==0 and x2 ==0 and y1==0 and y2==0:
        crop=frame
    else:
        crop = frame[x2:y2, x1:y1]
    return crop


def preview_gray_image(array):
    img_grey=cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_grey, cmap='gray')
    plt.show()


def contains_letters(s):
    return any(char.isalpha() for char in s)

def contains_numbers(s):
    return any(char.isdigit() for char in s)


def get_text_from_image(image_path):

    x1, x2, y1, y2, img = detect_number_plate(image_path)
    crop_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = list(reader.readtext(crop_gray))

    numbers= []
    for i in result:
        my_string = i[1]
        last_four_letters = my_string[-4:]
        if contains_numbers(last_four_letters):
            if contains_letters(last_four_letters):
                pass
            else:
                numbers.append(last_four_letters)

    res_img = cv2.putText(cv2.imread(image_path), str(numbers) , (x1, x2), cv2.FONT_HERSHEY_SIMPLEX, 3, (15,255,255), 5)
    plt.imshow(res_img, cmap='gray')
    plt.show()
    return numbers




if __name__ == '__main__':

    print(get_text_from_image(""))