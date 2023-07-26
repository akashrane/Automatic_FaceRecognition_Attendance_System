import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime


    #we import the images
path = 'Attendence'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

for cls in mylist:
    currImg = cv2.imread(f'{path}/{cls}')
    images.append(currImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


# we encoded the image
def Encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markingattendence(name):
    with open('attendecesheet.csv','r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')



encodelistknown = Encodings(images)
print('Encoding Done')

cap = cv2.VideoCapture(0)

while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        Facesincurrframe = face_recognition.face_locations(imgS)
        encodecurrframe = face_recognition.face_encodings(imgS,Facesincurrframe)

        for encodeface,faceloc in zip(encodecurrframe,Facesincurrframe):
            matches = face_recognition.compare_faces(encodelistknown,encodeface)
            facedis = face_recognition.face_distance(encodelistknown,encodeface)
            print(facedis)
            matchindex = np.argmin(facedis)

            if matches[matchindex]:
                name = classNames[matchindex].upper()
                print(name)
                y1,x2,y2,x1 = faceloc
                y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_ITALIC,2,(255,255,255),2)
                markingattendence(name)


        cv2.imshow('webcam',img)
        cv2.waitKey(1)

