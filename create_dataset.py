import cv2
import numpy as numpy

faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
stream = cv2.VideoCapture('http://192.168.1.4/mjpeg/1')
id = input('Masukkan id User:')
sampleNum = 0;
while(True): 
    ret,img =stream.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray);
    for (x,y,w,h) in faces:
        sampleNum = sampleNum+1;
        cv2.imwrite('dataSet/User.' +str(id)+"."+str(sampleNum)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y), (x+w,y+h),(0,255,0),2)
        cv2.waitKey(100);
        cv2.imshow("Face",img);
        cv2.waitKey(1);
    if(sampleNum>20):
        break;
stream.release()
cv2.destroyAllWindows()