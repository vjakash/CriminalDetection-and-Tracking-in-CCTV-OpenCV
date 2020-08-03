# facerec.py
import cv2, sys, numpy, os
import datetime
import urllib.request
import numpy as np
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
model =  cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
url="http://192.168.1.8:8080/shot.jpg"
cc,nc,c2c=0,0,0
while True:
    
    (_, im) = webcam.read()
    '''imgPath=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
    im1=cv2.imdecode(imgNp,-1)'''
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
            #print (names[prediction[0]])
            if names[prediction[0]]=="Non-Criminal":
                nc+=1
            elif names[prediction[0]]=="criminal":
                cc+=1
            elif names[prediction[0]]=="criminal-2":
                c2c+=1
            if names[prediction[0]]!="Non-Criminal":
                f=open("1.txt",'a')
                f.write('Printed string %s recorded at %s.\n' %(1, datetime.datetime.now()))

            cv2.putText(im,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    cv2.imshow('cam1', im)
    if nc!=0 or cc!=0 or c2c!=0:
        print(f"accuracy test for detecting criminal with {nc+cc+c2c} validation images")
        print(f"no.of non criminal detected:{nc}",f"no.of criminal detected:{cc}",f"no.of criminal-2 detected:{c2c}")
        print(f"accuracy{(cc/(nc+cc+c2c))*100}%")
    '''gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
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
            cv2.putText(im1,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im1,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    cv2.imshow('cam2', im1)'''
    key = cv2.waitKey(10)
    if key == 27:
        break
