from flask import Flask, flash, request, redirect, url_for, render_template, Response,jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS, cross_origin
from object_detector import detector
import base64


app=Flask(__name__)

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./"+fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

model = detector(filepath="file.jpg")


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')



class ClientApp:
    def __init__(self):
        self.filename = "file.jpg"
        # modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = detector(self.filename)


def run_inference(img_path='file.jpg'):
    # run inference using detectron2
    result_img = model.image_detection(img_path)

    # clean up
    try:
        os.remove(img_path)
    except:
        pass

    return result_img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        result = clApp.objectDetection.image_detection()

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"
    return jsonify(result)


if __name__ == '__main__':
    clApp = ClientApp()
    app.run()