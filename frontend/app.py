from flask import Flask,render_template,Response
import cv2
from util import *
from ultralytics import YOLO

model = YOLO("frontend\\artifacts\\detection\\best.pt")

app=Flask(__name__)
camera=cv2.VideoCapture(1)
camera2=cv2.VideoCapture(1)

def generate_frames():
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            cv2.imwrite("frame.jpg", frame)

            x1, x2, y1, y2, frame = detect_number_plate("frame.jpg")

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def crop_plate():
    while True:
        success,frame2=camera2.read()
        if not success:
            break
        else:
            cv2.imwrite("frame2.jpg", frame2)
            crop = number_plate_crop("frame2.jpg")
            # cv2.imwrite()
            ret,buffer=cv2.imencode('.jpg',crop)
            crop_frame=buffer.tobytes()

            yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + crop_frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main_video')
def main_video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plate_video')
def plate_video():
    return Response(crop_plate(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
