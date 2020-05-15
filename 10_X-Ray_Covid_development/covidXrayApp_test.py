'''
Python Script for testing of covidXrayApp
at terminal sus as:

$ python covidXrayApp_test.py

'''

# Import Libraries and Setup

import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Turn-off Info and warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Support Functions

def test_rx_image_for_Covid19_2(model, imagePath):
    img = cv2.imread(imagePath)
    img_out = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = model.predict(img)
    pred_neg = round(pred[0][1]*100)
    pred_pos = round(pred[0][0]*100)
    
    if np.argmax(pred, axis=1)[0] == 1:
        prediction = 'NEGATIVE'
        prob = pred_neg
    else:
        prediction = 'POSITIVE'
        prob = pred_pos

    cv2.imwrite('./Image_Prediction/Image_Prediction.png', img_out)
    return prediction, prob

# load model
covid_pneumo_model = load_model('./model/covid_pneumo_model.h5')

# ---------------------------------------------------------------
# Execute test

imagePath = './dataset_validation/covid_validation/6C94A287-C059-46A0-8600-AFB95F4727B7.jpeg'
prediction, prob = test_rx_image_for_Covid19_2(covid_pneumo_model, imagePath)
print (prediction, prob)
   