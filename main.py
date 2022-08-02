import cv2
import filter


img_folder = "images\\"
img_file = "will_test.png"
pattern = "images\\will.png"
img_save = "worked_on.jpg"
img_path = img_folder+img_file
img_path_save = img_folder+img_save


pattern = cv2.imread(pattern)
#pattern = cv2.cvtColor(pattern,cv2.COLOR_BGR2GRAY)
img = cv2.imread(img_path)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = filter.matchPattern(img,pattern)
cv2.imshow("img",img)
cv2.imwrite(img_path_save,img)




