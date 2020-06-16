from  red_waterline.text_detection import bloom_debug, bloom2_debug, \
    separate_color

from tools.imutils import cv_imread
import cv2
import numpy as np
from red_waterline.union_find import UnionFind
import os

import matplotlib.pyplot as plt




# 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass

def hsv_trackbar():
    # 1 正常晕染
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161831.jpg'
    # 2 粗线晕染
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161817.jpg'
    # 3 正常1
    imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092349.jpg'
    # 3 正常2
    # imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092610.jpg'

    img = cv_imread(imagePath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)

    # 创建RGB三个滑动条
    cv2.createTrackbar('H1', 'image', 170, 180, nothing)
    cv2.createTrackbar('H2', 'image', 180, 180, nothing)

    cv2.createTrackbar('S1', 'image', 60, 255, nothing)
    cv2.createTrackbar('S2', 'image', 115, 255, nothing)

    cv2.createTrackbar('V1', 'image', 70, 255, nothing)
    cv2.createTrackbar('V2', 'image', 120, 255, nothing)

    res = img.copy()
    while True:

        cv2.imshow('image', np.vstack((img, res)))
        if cv2.waitKey(1) == 27:
            break

        # 获取滑块的值
        # val = cv2.getTrackbarPos('R', 'image')
        H1 = cv2.getTrackbarPos('H1', 'image')
        H2 = cv2.getTrackbarPos('H2', 'image')
        S1 = cv2.getTrackbarPos('S1', 'image')
        S2 = cv2.getTrackbarPos('S2', 'image')
        V1 = cv2.getTrackbarPos('V1', 'image')
        V2 = cv2.getTrackbarPos('V2', 'image')

        lower_hsv = np.array([H1, S1, V1])  # 提取颜色的低值
        high_hsv = np.array([H2, S2, V2])  # 提取颜色的高值
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)

        res = cv2.bitwise_and(img, img, mask=mask)


def gary_trackbar():
    # 1 正常晕染 UpG_161405 UpG_161410 UpG_161734
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161903.jpg'
    # 2 粗线晕染
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161817.jpg'
    # 3 正常1
    # imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092349.jpg'
    # 3 正常2
    imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092629.jpg'

    img = cv_imread(imagePath)
    mask_red, redhsv = separate_color(img, color='red')
    # cv2.imshow('ori', img)
    # cv2.imshow('tiqutu', redhsv)

    h, w, _ = img.shape
    for i in range(0, h):
        for j in range(0, w):
            a, b, c = redhsv[i][j]
            if a == 0 and b == 0 and c==0:
                redhsv[i][j] = [255,255,255]

    cv2.imshow('bai', redhsv)

    res_ori = cv2.cvtColor(redhsv, cv2.COLOR_BGR2GRAY)
    print(res_ori.shape)
    vals = []
    for i in range(0, h):
        for j in range(0, w):
            if res_ori[i][j] != 255:
                vals.append(res_ori[i][j])
    vals = np.array(vals)

    # plt.hist(vals,255)
    # plt.show()

    line_gray_mean = np.mean(vals)
    print('红水线灰度化均值:', line_gray_mean)

    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('G', 'image', 88, 255, nothing)
    cv2.createTrackbar('TH', 'image', 10, 40, nothing)



    while True:
        if cv2.waitKey(1) == 27:
            break

        G = cv2.getTrackbarPos('G', 'image')
        TH = cv2.getTrackbarPos('TH', 'image')

        _, res = cv2.threshold(res_ori,G,255,cv2.THRESH_BINARY)

        ser = cv2.bitwise_not(res)

        # 以(30,2)长度的滑动窗口，滑动，如果有校像素超过3/4 28? ,认为是深色红水线
        # 中值滤波呢，不太行
        # midblur = cv2.medianBlur(ser, 3)

        valid = []
        blank = np.zeros((h, w), dtype=np.uint8)  # 黑板

        kh = 2
        kw = 20
        for i in range(0,h-kh):
            for j in range(0,w-kw):
                temp = ser[i:i+kh, j:j+kw]
                white = np.sum(temp==255)
                if white >= TH:
                    blank[i:i + kh, j:j + kw] = 255

        cv2.imshow('moban', blank)
        cv2.imshow('image', np.vstack((res_ori, ser)))


def a(x,y):
    return x+y



if __name__ == '__main__':
    # bloom2_debug()
    # gary_trackbar()
    c = a(1,y=2)
    print(c)










