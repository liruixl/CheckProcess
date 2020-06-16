
import os
import  cv2
import numpy as np

work_dir = r"C:\Users\lirui\Desktop\temp\check"
img_path = os.path.join(work_dir, '092339_Upir.jpg')

img_arr = cv2.imread(img_path)  # BGR
w, h = img_arr.shape[:2]

print(img_arr.shape[:2])

for i in range(w):
    for j in range(h):
        if img_arr[i][j][0] < 150 and img_arr[i][j][1] < 150 and img_arr[i][j][2] < 150:
            pass
        else:
            img_arr[i][j] = [255, 255, 255]

save_name = r'C:\Users\lirui\Desktop\temp\gray_like.jpg'
cv2.imwrite(save_name, img_arr)





