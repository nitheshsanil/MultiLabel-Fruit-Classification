#Import necessary libraries
from flask import Flask, render_template, Response
# from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
#Initialize the Flask app
app = Flask(__name__)
# model = load_model('keras_model.h5')


interpreter = tf.lite.Interpreter(model_path="model_fruit.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']


camera = cv2.VideoCapture(0)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            classes = {0 : "apple_braeburn", 1: "apple_red_delicious", 2: "banana", 3: "grapes", 4: "others"}
            # object = []
            # image = Image.fromarray(frame).convert('RGB')
            # ##Converting image into tensor
            # image_tensor = read_tensor_from_readed_frame(image ,224, 224)
            input_frame = cv2.resize(frame, (224,224))
            input_frame = ((np.expand_dims(input_frame, 0).astype(np.float32))/127.0)-1
            #Test model
            interpreter.set_tensor(input_details[0]['index'], input_frame)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = output_data[0]*100
            print(output_data)
            object_out = classes[np.argmax(output_data)]
            frame = cv2.putText(frame, f'Object(s) detected: {object_out}', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1) 
            # cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            ret, buffer = cv2.imencode('.jpg', frame)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
