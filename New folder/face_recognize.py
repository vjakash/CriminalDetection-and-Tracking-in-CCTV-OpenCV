# facerec.py
import cv2, sys, numpy, os
import urllib.request
import numpy as np
import time
import os
import glob
import imutils
import sys
import datetime



#port = serial.Serial("com",9600)

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

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

a1=0;
a2=0;
a3=0;
ip_cam=["192.168.1.6:8080"]
while True:
    for ip in range(len(ip_cam)):
        url="http://"+ip_cam[ip]+"/shot.jpg"
        imgPath=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
        img=cv2.imdecode(imgNp,-1)
        img=imutils.resize(img,width=300)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            #Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if prediction[1]<500:
                #port.write('B')
                print (names[prediction[0]])
                cv2.putText(img,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                if names[prediction[0]]=='xxxx':
                    a1=a1+1
                    if a1==5:
                        print("Criminal detected")
                
            else:
                cv2.putText(img,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        cv2.imshow('OpenCV', img)
        key = cv2.waitKey(10)
        cv2.destroyAllWindows()
    
       
