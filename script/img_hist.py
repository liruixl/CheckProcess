
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'../img/fiber_red_2.jpg')

#  H [0, 179] S [0, 255] V [0, 255]
#1  H [0,30]/[160,180] S [0,75]    V [100, 180]
#2  H [0,30]/[160.180] S [0,40/80] V [100,170/180]
#in H [0,30]/[160,180] S [0.20] V [140,180]/[150,170]

# yiingzhang
# H([0,10] and [175,180]) S[180,230]  V [75,90/95]

# cu_line 异常晕染变粗的红水线
# H [0,10] and [170,180]  S[20, 130] or [60,115] V [70, 120]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower_hsv = np.array([165, 43, 46])  # 提取颜色的低值
# high_hsv = np.array([180, 255, 255])  # 提取颜色的高值
# mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
# lower_hsv = np.array([0, 43, 46])  # 提取颜色的低值
# high_hsv = np.array([10, 255, 255])
# mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv))

H = hsv[:,:,0]
S = hsv[:,:,1]
V = hsv[:,:,2]

plt.hist(H.ravel(), 180)
plt.show()
plt.hist(S.ravel(), 256)
plt.show()
plt.hist(V.ravel(), 256)
plt.show()


# 坐标太小看不清
# plt.subplot(2,2,1)
# plt.hist(H.ravel(), 180)
# plt.subplot(2,2,2)
# plt.hist(S.ravel(), 256)
# plt.subplot(2,2,3)
# plt.hist(V.ravel(), 256)
# plt.show()