import cv2
import numpy as np

img = cv2.imread("images\\crops.jpg")
#img = np.array([[0,2],[1,2]])

print(img[0,0])

print(img[0,0]!=0)

if any(img[0,0])!=0:
    print("pixel")