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
    payload = "sender_id=CCTV-Criminal_Detector&message="+msg+"&language=english&route=p&numbers=8870631234,7338987648"
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
cam1= cv2.VideoCapture(0)
cam2="http://192.168.1.8:7070/shot.jpg"
cam3="http://192.168.1.6:7070/shot.jpg"
cam=[cam1,cam2,cam3]
graph={cam1:[cam2],cam2:[],cam3:[]}

rot=""
count1,check1=0,0
count2,check2=0,0
count3,check3=0,0
loc1="https://goo.gl/maps/rmtQR4boAP9vPzaZ8"
while True:
    
    (_, im) = cam1.read()
    imgPath=urllib.request.urlopen(cam2)
    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
    im1=cv2.imdecode(imgNp,-1)
    '''imgPath=urllib.request.urlopen(cam3)
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
            if names[prediction[0]]!="Non-Criminal" :
                check1+=1
                if check1>8:
                    s1=names[prediction[0]]+" recorded at "+ str( datetime.datetime.now())+" in "+"cam-1"+" ||   Location of the cam:"+loc1
                    #sendsms(s1)
                    print(s1)
                    count1=1
                    rot=cam1
                    break
                     
            cv2.putText(im,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    if(count1==1):
        cv2.destroyAllWindows();
        break
    cv2.imshow('cam1', im)
    ###################################################################################################################################
    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
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
            if names[prediction[0]]!="Non-Criminal" :
                check2+=1
                if check2>8:
                    s2=names[prediction[0]]+" recorded at "+str( datetime.datetime.now())+" in "+"cam-2"+" ||   Location of the cam:"+loc1
                    #sendsms(s2)
                    count2=1
                    rot=cam2
                    break

                     
            cv2.putText(im1,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im1,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    if(count2==1):
        cv2.destroyAllWindows();
        break
    cv2.namedWindow("Cam2", cv2.WINDOW_NORMAL)        
    cv2.resizeWindow('Cam2', 650, 500)                   
    cv2.imshow("Cam2", im1)                 

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
            if names[prediction[0]]!="Non-Criminal" :
                check3+=1
                if check3>8:
                    s3=names[prediction[0]]+" recorded at "+str( datetime.datetime.now())+" in "+"cam-3"+" ||   Location of the cam:"+loc1
                    #sendsms(s3)
                    count3=1
                    rot=cam3
                    break
                
                   
            cv2.putText(im2,names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        else:
            cv2.putText(im2,'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
    if(count3==1):
        cv2.destroyAllWindows();
        break
    #cv2.imshow('cam3', im2)
    cv2.namedWindow("Cam3", cv2.WINDOW_NORMAL)       
    cv2.resizeWindow('Cam3', 650, 500)                   
    cv2.imshow("Cam3", im2)  '''  
    key = cv2.waitKey(10)
    if key == 27:
        break
##############################################################################################################################################
def perimeter(root):
    s=set()
    imnew=[]
    ind=[]
    checkn=0
    #s.add(root)
    for i in graph[root]:
        s.add(i)
        for j in graph[i]:
            s.add(j)
    while True:
        imnew=[]

        for i in s:
            if i==cam1:
                ind.append(cam.index(i))
                imnew.append( cam1.read())
            else:
                ind.append(cam.index(i))
                imgPath=urllib.request.urlopen(i)
                imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
                imnew.append(cv2.imdecode(imgNp,-1))

        for i in range(len(imnew)):
            
            gray = cv2.cvtColor(imnew[i], cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(imnew[i],(x,y),(x+w,y+h),(255,255,0),2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                #Try to recognize the face
                prediction = model.predict(face_resize)
                cv2.rectangle(imnew[i], (x, y), (x + w, y + h), (0, 255, 0), 3)

                if prediction[1]<500:
                    print (names[prediction[0]])
                    if names[prediction[0]]!="Non-Criminal" :
                        checkn+=1
                        if checkn>8:
                            s1=names[prediction[0]]+" recorded at "+ str( datetime.datetime.now())+" in "+"cam-"+str(ind[i]+1)+" ||   Location of the cam:"+loc1
                            #sendsms(s1)
                            print(s1)
                            rot=list(s)[i]
                            print(rot)
                            cv2.destroyAllWindows();
                            perimeter(rot)
                        
                         
                    cv2.putText(imnew[i],names[prediction[0]],(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                else:
                    cv2.putText(imnew[i],'Scanning',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            win="cam"+str(ind[i]+1)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)       
            cv2.resizeWindow(win, 650, 500)                   
            cv2.imshow(win, imnew[i])
            key = cv2.waitKey(10)
            if key == 27:
                break
        

perimeter(rot)
