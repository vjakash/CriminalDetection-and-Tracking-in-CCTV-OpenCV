import time 
import requests
import json
import cv2, sys, numpy, os
import datetime
import urllib.request
import numpy as np
cam1="http://192.168.1.4:8080/shot.jpg"
cam2="http://192.168.1.8:8080/shot.jpg"
cam3="http://192.168.1.6:8080/shot.jpg"
cam=[cam1,cam2,cam3]
s=set()
graph={cam1:[cam2,cam3],cam2:[cam1,cam2],cam3:[]}
def peri(root):
    for i in graph[root]:
        s.add(i)
        for j in graph[i]:
            s.add(j)
    sorted(s)
    for i in s:print(i)
    print(s)
    while True:
        for i in s:
            if i==cam1:
                ind.append(cam.index(i))
               # imnew.append( cam1.read())
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
                        s1=names[prediction[0]]+" recorded at "+ str( datetime.datetime.now())+" in "+"cam-"+str(ind[i]+1)+" ||   Location of the cam:"+loc1
                       # sendsms(s1)
                        print(s)
                        rot=s[i]
                        
                         
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

imnew=[]
ind=[]
peri(cam1)

