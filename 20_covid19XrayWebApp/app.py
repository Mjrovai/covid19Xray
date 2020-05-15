
# Import Libraries and setup
from flask import Flask, request, g, redirect, url_for, flash, render_template, make_response
import requests
import os
import datetime
#import random
from pathlib import Path
import shutil
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import cv2
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Functions
def test_rx_image_for_Covid19(model, imagePath, filename):
    img = cv2.imread(imagePath)
    img_out = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = model.predict(img)
    pred_neg = int(round(pred[0][1]*100))
    pred_pos = int(round(pred[0][0]*100))

    if np.argmax(pred, axis=1)[0] == 1:
        prediction = 'NEGATIVE'
        prob = pred_neg
    else:
        prediction = 'POSITIVE'
        prob = pred_pos

    img_pred_name =  prediction+'_Prob_'+str(prob)+'_Name_'+filename+'.png'
    cv2.imwrite('static/xray_analisys/'+img_pred_name, img_out)
    cv2.imwrite('static/Image_Prediction.png', img_out)
    print
    return prediction, prob


UPLOAD_FOLDER = os.path.join('static', 'xray_img')
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG'])

covid_pneumo_model = load_model('./model/covid_pneumo_model.h5')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

prediction=' '
confidence=0
filename='Image_No_Pred_MJRoBot.png'
image_name = filename


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def hello_world():
	return render_template('index.html', )

@app.route("/query", methods=["POST"])
def query():
    if request.method == 'POST':
        # RECIBIR DATA DEL POST
        if 'file' not in request.files:
            return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        file = request.files['file']
        # image_data = file.read()
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
        if file and allowed_file(file.filename):

            filename = str(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            image_name = filename

            # detection covid
            try:
                prediction, prob = test_rx_image_for_Covid19(covid_pneumo_model, img_path, filename)
                return render_template('index.html', prediction=prediction, confidence=prob, filename=image_name, xray_image=img_path)
            except:
                return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename=image_name, xray_image=img_path)
        else:
            return render_template('index.html', name='FILE NOT ALOWED', confidence=0, filename=image_name, xray_image=img_path)

# No caching at all for API endpoints.

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run()
