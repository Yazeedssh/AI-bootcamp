import cv2
import numpy as np 
import matplotlib.pyplot as plt 

'''This code for detecting objects corners'''
imge = cv2.imread("TR.JFIF")
gray = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

des = cv2.cornerHarris(gray,2,5,0.07)
des = cv2.dilate(des, None)
imge[des>0.01 * des.max()]=[255,0,0]

plt.imshow(imge)
plt.waitforbuttonpress()