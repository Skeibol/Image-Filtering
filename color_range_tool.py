#taken from and modified: https://medium.com/programming-fever/how-to-find-hsv-range-of-an-object-for-computer-vision-applications-254a8eb039fc
IMAGE_PATH = r"images\crops.jpg"


import cv2
import numpy as np
import time

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - X", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - Y", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - Z", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - X", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - Y", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - Z", "Trackbars", 255, 255, nothing)
 
while True:
    frame = cv2.imread(IMAGE_PATH)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - X", "Trackbars")
    l_s = cv2.getTrackbarPos("L - Y", "Trackbars")
    l_v = cv2.getTrackbarPos("L - Z", "Trackbars")
    u_h = cv2.getTrackbarPos("U - X", "Trackbars")
    u_s = cv2.getTrackbarPos("U - Y", "Trackbars")
    u_v = cv2.getTrackbarPos("U - Z", "Trackbars")
 
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_range, upper_range)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((mask_3,frame,res))

    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.5,fy=0.5))
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    if key == ord('s'):    
        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)
        
        # Also save this array as penval.npy
        np.save('hsv_value',thearray)
        break
    
# Release the camera & destroy the windows.    
cv2.destroyAllWindows()