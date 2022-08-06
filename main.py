import cv2
import filter
import numpy as np

img_folder = "images\\"
img_file = "crops.jpg"
pattern = "images\\face_pattern.jpg"
img_save = "worked_on.jpg"
img_path = img_folder+img_file
img_path_save = img_folder+img_save


# pattern = cv2.imread(pattern)
# pattern = cv2.cvtColor(pattern,cv2.COLOR_BGR2GRAY)
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask_min =(22,0,0)
mask_max = (74,255,255)

img_cv = cv2.inRange(img,mask_min,mask_max)
img = filter.mask(img,mask_min,mask_max)
img = filter.dilate(img,3)
# cv2.imshow("img",img)
cv2.imwrite(img_path_save,img)
cv2.imwrite("images\\worked_on_cv.jpg",img_cv)




