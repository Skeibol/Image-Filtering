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
mask = cv2.imread("images\\mask.jpg")
print(mask.shape,img.shape)
img = filter.bitwise_and(img,mask)
# cv2.imshow("img",img)
cv2.imwrite(img_path_save,img)




