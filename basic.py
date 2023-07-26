import face_recognition
import cv2
import numpy as np

imgElon = face_recognition.load_image_file('Images Basics/Elon Musk.jpeg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images Basics/Elon musk Test.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

Faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(Faceloc[3],Faceloc[0]),(Faceloc[1],Faceloc[2]),(0, 255, 255),4)

FacelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(FacelocTest[3],FacelocTest[0]),(FacelocTest[1],FacelocTest[2]),(0, 255, 255),4)

results = face_recognition.compare_faces([encodeElon],encodeTest)
FaceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,FaceDis)
cv2.putText(imgTest,f'{results} {round(FaceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0, 255, 255),4)

cv2.imshow('Elon Dada',imgElon)
cv2.imshow('Elon Dada Test Photo',imgTest)

cv2.waitKey(0)