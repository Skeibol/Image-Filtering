import cv2
import filter


img_folder = "images\\"
img_file = "face.jpg"
pattern = "images\\face_pattern.jpg"
img_save = "worked_on.jpg"
img_path = img_folder+img_file
img_path_save = img_folder+img_save


# pattern = cv2.imread(pattern)
# pattern = cv2.cvtColor(pattern,cv2.COLOR_BGR2GRAY)
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask_min =[170,245,245]
mask_max = [190,255,255]

img = filter.mask(img,mask_min,mask_max,inverse=True)
img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
# cv2.imshow("img",img)
cv2.imwrite(img_path_save,img)




