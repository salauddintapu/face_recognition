import cv2
import json
import base64
import warnings
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
from recognition import RECOGNITION

warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)

face= RECOGNITION()

def convert_to_im_array(data):
    arr = base64.b64decode(data)
    img_arr = np.frombuffer(arr, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(image):
    res, frame = cv2.imencode('.jpg', image)    # from image to binary buffer
    data = base64.b64encode(frame) 
    return data
        
@app.route('/id_info', methods=['POST', 'GET'])
def get_predictions():
    if request.method == 'POST':
        try:
            data = json.loads(request.data.decode('utf-8'))
            file = data['file']
            img_bytes = file
            img_bytes = convert_to_im_array(img_bytes)
            img = np.asarray(img_bytes)
            print('=======================', img.shape)

            try:
                res = face.face_verify(img)
                return jsonify(res)
            
            except Exception as e:
                return jsonify({'result': 'error during prediction', 'error': e})

        except Exception as e:
            # print(e)
            return jsonify({'result': 'error during prediction', 'error': e})


if __name__ == '__main__':
    print("SERVER STARTED")
    app.run(host="localhost",port=5000,debug=True)
