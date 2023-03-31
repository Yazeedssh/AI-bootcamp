import cv2
import numpy as np
import requests
from PIL import Image

'''Image request using requests libarary and image processing '''
img = Image.open(requests.get('https://a57.foxnews.com/media.foxbusiness.com/BrightCove/854081161001/201805/2879/931/524/854081161001_5782482890001_5782477388001-vs.jpg',stream=True).raw)
img = img.resize((450,250))
img_arr= np.array(img)
gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
Image.fromarray(blur)
dilated = cv2.dilate(blur,np.ones((3,3)))
Image.fromarray(dilated)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE,kernel)
Image.fromarray(closing)

'''calling the trained data'''
car_src = 'haarcascade_car.xml'
car = cv2.CascadeClassifier(car_src)
cars = car.detectMultiScale(closing, 1.1, 1)

'''Creating the rectangle on the cars'''
cnt = 0 
for (x,y,w,h) in cars:
    cv2.rectangle(img_arr, (x,y),(x+w,y+h),(255,0,0),2)
    cnt += 1
    print(cnt,"CARS")
    Image.fromarray(img_arr)

'''Show the image'''
cv2.imshow('image',img_arr)
cv2.waitKey(0)
cv2.destroyAllWindows()
