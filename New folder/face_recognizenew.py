# facerec.py
import cv2, sys, numpy, os
import urllib.request
import numpy as np
import time
import os
import glob

import sys
import datetime




#port = serial.Serial("com",9600)

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'  #All the faces data will be present this folder

####sub_data = 'hai'  #These are sub data sets of folder, for my faces I've used my name




print('Training...')
# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)

##with open("1.txt", mode='a') as file:
webcam = cv2.VideoCapture(0)
a1=0;
a2=0;
a3=0;
url="http://192.168.43.1:8080/shot.jpg"
url1="http://192.168.43.135:8080/shot.jpg"
#url2="http://192.168.43.1:8080/shot.jpg"
count = 1
while True:
##        (_, im) = webcam.read()
        imgPath=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
        im=cv2.imdecode(imgNp,-1)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            #Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1]<500:
                print (names[prediction[0]])
                cv2.putText(im,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                if names[prediction[0]]=='criminal':
                    a1=a1+1
                    if a1==5:
                        sub_data = 'criminal'
                        path = os.path.join(datasets, sub_data)
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        (width, height) = (130, 100)    # defining the size of images 
                        cv2.imwrite('%s/%s.png' % (path,count), face_resize)
                        imgPath1=urllib.request.urlopen(url1)
                        imgNp1=np.array(bytearray(imgPath1.read()),dtype=np.uint8)
                        im1=cv2.imdecode(imgNp1,-1)
##                        imgPath2=urllib.urlopen(url2)
##                        imgNp2=np.array(bytearray(imgPath2.read()),dtype=np.uint8)
##                        im2=cv2.imdecode(imgNp2,-1)
                        count += 1
                        a1=a1+1;
                        cv2.imshow('Cam2', im1)
##                        cv2.imshow('Cam3', im2)

                
            else:
                cv2.putText(im,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        cv2.imshow('Cam1', im)
       
        key = cv2.waitKey(10)
    
       
