import cv2
import filter


img_folder = "images\\"
img_file = "face_female.jpg"
pattern = "images\\face_pattern.jpg"
img_save = "worked_on.jpg"
img_path = img_folder+img_file
img_path_save = img_folder+img_save

img = cv2.imread(img_path)
img = filter.dilate(img,7)

cv2.imshow("img",img)
cv2.imwrite(img_path_save,img)




