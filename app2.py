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
            classes = {1 :"apple_red_delicious", 0: "apple_braeburn", 2: "banana", 3: "grapes", 4:"others"}
            # object = []
            # image = Image.fromarray(frame).convert('RGB')
            # ##Converting image into tensor
            # image_tensor = read_tensor_from_readed_frame(image ,224, 224)
            input_frame = cv2.resize(frame, (224,224))
            input_frame = ((np.expand_dims(input_frame, 0).astype(np.float32))/127.0)-1
            # input_frame = input_frame * 255
            # input_frame = input_frame.astype(np.uint8)
            #Test model
            interpreter.set_tensor(input_details[0]['index'], input_frame)
            interpreter.invoke()
            
            res = []
            for i in range(4):
                res.append(interpreter.get_tensor(output_details[i]['index']))
                
            result_index = []
            for j in res[0][0]:
                if j>=0.5:
                    result_index.append(list(res[0][0]).index(j))
            		
            result = {}
            for idx in result_index:
                result[classes[res[3][0][idx]]] = res[0][0][idx]
            
            print(result)
            # frame = cv2.putText(frame, f'Object(s) detected: {result.keys}', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 1) 
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




    '''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''


"""
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('<IMAGE_PATH>')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)


"""
