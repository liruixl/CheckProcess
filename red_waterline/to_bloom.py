import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tools.imutils import cv_imread, separate_color


def bloom_d(img, debug=False):
    print('红水线尺寸：', img.shape)
    img_bgr = img.copy()  # 用于绘制
    mask_red, redhsv = separate_color(img_bgr, color='red')
    if debug:
        cv2.namedWindow('redhsv', cv2.WINDOW_NORMAL)
        cv2.imshow('redhsv', redhsv)
    mask_yz, yingzhang = separate_color(img_bgr, color='yingzhang')
    mask_yz = cv2.bitwise_not(mask_yz)

    if debug:
        cv2.imshow('yingzhang', yingzhang)


    # redline = cv2.bitwise_and(redhsv, redhsv, mask=mask_yz)
    # ==================================================
    redline = redhsv.copy()
    gray = cv2.cvtColor(redline, cv2.COLOR_BGR2GRAY)  # 打算用灰度图去判断深色的晕染线。。
    # ==================================================

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 1*W*10 <= x <= 2*W*10  正常红水线

    cnt = 0
    h, w = binary.shape
    for i in range(h):
        for j in range(w):
            if binary[i][j] > 10:
                cnt += 1

    print('range should be in ', w*10, 2*w*10)
    print('红水线比例：', cnt, '/', h*w)


def tidu():
    img = cv2.imread('../img/redline/gratch_irtr_1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('irtr gray', img)

    v1 = cv2.Canny(img, 80, 250)  # minValue取值较小，边界点检测较多,maxValue越大，标准越高，检测点越少
    v2 = cv2.Canny(img, 50, 100)

    res = np.hstack((v1, v2))
    cv2.imshow("res", res)
    cv2.waitKey(0)


if __name__ == '__main__':

    tidu()

    # imagePath = r'../img/redline/bloom_d_1.jpg'
    # # imagePath = r'../img/redline/normal_1.jpg'
    # img = cv_imread(imagePath)
    # bloom_d(img, True)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()