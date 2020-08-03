# facerec.py
import time 
import requests
import json
import cv2, sys, numpy, os
import datetime
import urllib.request
import numpy as np
#####################################################################################################
def sendsms(msg):
    url = "https://www.fast2sms.com/dev/bulk"
    payload = "sender_id=FSTSMS&message="+msg+"&language=english&route=p&numbers=8870631234,7338987648"
    headers = {
    'authorization': "j39GYkoNaKLegEOTtxSIvyfupC4brAUWXHFJlmqh7Q1RB8sci2dLy2Ge5pNWb6CIORh7rV1zYcwUxfuo",
    'Content-Type': "application/x-www-form-urlencoded",
    'Cache-Control': "no-cache",
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    print(response.text)
#######################################################################################################
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
#url1="http://192.168.1.6:8080/shot.jpg"
count1=0
count2=0
count3=0
loc1="https://goo.gl/maps/rmtQR4boAP9vPzaZ8"
acc=[]
acc1=[]
while True:
    
    (_, im) = webcam.read()
    '''
    imgPath=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
    im1=cv2.imdecode(imgNp,-1)'''
    '''imgPath=urllib.request.urlopen(url1)
    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
    im2=cv2.imdecode(imgNp,-1)'''
    ###############################################################################################################################
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
            
            if names[prediction[0]]!="Non-Criminal":
                name=names[prediction[0]]
                '''if(len(acc)==0):
                    acc.append(names[prediction[0]])
                elif names[prediction[0]] in acc:
                    acc.append(names[prediction[0]])
                else:
                    acc1.append(names[prediction[0]])
                print(acc)
                print(acc1)
                if len(acc)>=len(acc1):
                    name=acc[0]
                else:
                    name=acc1[0]
                print(name)'''
                s1=name+" recorded at "+ str( datetime.datetime.now())+" in "+"cam-1"+"/nLocation of the cam:"+loc1
               #sendsms(s1)
                count1=1
                with open("1.txt", mode='a') as file:
                    file.write('%s recorded at %s in %s.\n' % (name, datetime.datetime.now(),"cam-1"))
                cv2.putText(im,name,(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
                cv2.putText(im,'Non-Criminal',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    cv2.imshow('cam1', im)
    ###################################################################################################################################
    '''gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im1,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        #Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im1, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1]<500:
            print (names[prediction[0]])
            if names[prediction[0]]!="Non-Criminal" and count2!=1:
                s2=names[prediction[0]]+" recorded at "+str( datetime.datetime.now())+" in "+"cam-2"+"/nLocation of the cam:"+loc1
                #sendsms(s2)
                count2=1
                with open("1.txt", mode='a') as file:
                    file.write('%s recorded at %s in %s.\n' % (names[prediction[0]], datetime.datetime.now(),"cam-2"))
                     
            cv2.putText(im1,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im1,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    cv2.namedWindow("Cam2", cv2.WINDOW_NORMAL)        
    cv2.resizeWindow('Cam2', 650, 500)                   
    cv2.imshow("Cam2", im1)                 
    '''
    ####################################################################################################################################
    '''gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im2,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        #Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1]<500:
            print (names[prediction[0]])
            if names[prediction[0]]!="Non-Criminal" and count3!=1:
                s3=names[prediction[0]]+" recorded at "+str( datetime.datetime.now())+" in "+"cam-3"+"/nLocation of the cam:"+loc1
                #sendsms(s3)
                count3=1
                with open("1.txt", mode='a') as file:
                    file.write('%s recorded at %s in %s.\n' % (names[prediction[0]], datetime.datetime.now(),"cam-3"))
                   
            cv2.putText(im2,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im2,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    #cv2.imshow('cam3', im2)
    cv2.namedWindow("Cam3", cv2.WINDOW_NORMAL)       
    cv2.resizeWindow('Cam3', 650, 500)                   
    cv2.imshow("Cam3", im2)    '''
    key = cv2.waitKey(10)
    if key == 27:
        break
