import cv2
import numpy as np 
import matplotlib.pyplot as plt

'''This code is for detecting corners of the letters in the text'''
image = cv2.imread("ma_name.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image),plt.show() #to show the image and the plot

corner = cv2.goodFeaturesToTrack(gray,30,0.01,10)
corner = np.int32(corner)

'''Specify the dots space, diameter, color'''
for i in corner:
    x,y = i.ravel()
    cv2.circle(image, (x,y), 3, 255, -1)

'''To show the image'''
plt.imshow(image)
plt.waitforbuttonpress()