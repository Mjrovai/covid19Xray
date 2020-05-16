# Covid-19 Xray Detection and Web-App development

<img src="https://github.com/Mjrovai/covid19Xray/blob/master/10_X-Ray_Covid_development/images/portada.png"/>
Developed by M.Rovai @ May, 12 2020<br>

## Disclaimer

The deployment of an automatic COVID-19 detection is for educational purposes only. It is not meant to be a reliable, highly accurate COVID-19 diagnosis system, nor has it been professionally or academically vetted.

## Introduction

**Inspiration**

The inspiration of this project, was to understand and create a didactic proof of concept of the work "[XRayCovid-19](http://tools.atislabs.com.br/covid)" developed by UFRRJ (Universidade Federal Rural do Rio de Janiero). **XRayCovid-19** is an ongoing project that uses Artificial Intelligence to assist the health system in the COVID-19 diagnostic process. It is characterized by easy use; efficiency in response time and effectiveness in the result.

**Why X-rays?**

There have been promising efforts to apply machine learning to aid in the diagnosis of COVID-19 based on CT scans. Despite the success of these methods, the fact remains that COVID-19 is an infection that is likely to be experienced by communities of all sizes. X-rays are inexpensive and quick to perform; therefore, they are more accessible to healthcare providers working in smaller and/or remote regions. 

**Thanks**

This work was developed using TensorFlow and Keras, based on the great tutorial published by [Dr. Adrian Rosebrock](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/). Also, I would like to thanks Nell Trevor that, also based on Dr. Rosebrock's work, provided an endpoint idea, where the resultant model could be tested: [Covid-19 predictor API](http://coviddetector.pythonanywhere.com)

## Papers

- [[1] 2020 Chowdhury et al - Can AI help in screening Viral and COVID-19 pneumonia?](https://arxiv.org/pdf/2003.13145.pdf)
- [[2] 2020 Hall et all - Finding COVID-19 from Chest X-rays using Deep Learning on a Small Dataset](https://arxiv.org/pdf/2004.02060.pdf)
- [[3] 2020 COVID-19 Screening on Chest X-ray Images Using Deep Learning based Anomaly Detection](https://arxiv.org/pdf/2003.12338.pdf)

## Datasets
**Dataset 1: COVID-19 image data collection** <br>
Joseph Paul Cohen and Paul Morrison and Lan Dao
COVID-19 image data collection, arXiv:2003.11597, 2020

Project Summary: To build a public open dataset of chest X-ray and CT images of patients which are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS.). Data will be collected from public sources as well as through indirect collection from hospitals and physicians. This project is approved by the University of Montreal's Ethics Committee #CERSES-20-058-D

All images and data will be released publicly in this [GitHub repo](https://github.com/ieee8023/covid-chestxray-dataset).

**Dataset 2: Chest X-Ray Images (Pneumonia)**<br>
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2 http://dx.doi.org/10.17632/rscbjbr9sj.2 

Dataset of validated OCT and Chest X-Ray images described and analyzed in "Deep learning-based classification and referral of treatable human diseases". The Images are split into a training set and a testing set of independent patients. Images are split into 2 directories: PNEUMONIA, and NORMAL.

Data file: https://data.mendeley.com/datasets/rscbjbr9sj/2
