import cv2
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import  Counter


def bgr_avg(img):
    b = np.mean(img[:,:,0])
    g = np.mean(img[:,:,1])
    r = np.mean(img[:,:,2])
    return [b,g,r]


def calc_diff(pixel, bg_color):
    # 计算pixel与背景的离差平方和，作为当前像素点与背景相似程度的度量
    return (pixel[0]-bg_color[0])**2 + (pixel[1]-bg_color[1])**2 + (pixel[2]-bg_color[2])**2


def _distance_bg(img, bg_color, threshold):
    h, w = img.shape[:2]

    for i in range(h):
        for j in range(w):
            if calc_diff(img[i][j], bg_color) < threshold:
                # 若果logo[i][j]为背景，将其颜色设为白色，且完全透明f
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
    return img


# 最大类间方差确定阈值 输入灰度图
def _var_max_interclass(img, calc_zero=True):
    h, w = img.shape[:2]
    mean = np.mean(img)

    counts = np.zeros((256,), dtype=int)
    prob = np.zeros((256,), dtype=float)

    count_map = Counter(img.ravel())
    for pix_val, count in count_map.items():
        if not calc_zero and pix_val == 0:
            continue
        counts[pix_val] = count
    prob = counts / (h*w)

    ks = [i for i in range(90,160)]
    maxvar = 0
    threshold = 0

    for k in ks:
        p1 = np.sum(prob[:k])
        p2 = 1-p1

        mean1 = [i*prob[i] for i in range(k)]
        # mean1 = np.sum(mean1)/p1
        mean1 = np.sum(mean1)


        # mean11 = [i * counts[i] for i in range(k)]
        # sum11 = np.sum(counts[:k])
        # mean11 = np.sum(mean11) / sum11

        var = (mean*p1 - mean1)**2 / (p1*p2)

        if var > maxvar:
            maxvar = var
            threshold = k

    return threshold


if __name__ == '__main__':

    work_dir = r'C:\Users\lirui\Desktop\temp\check\redline'
    temp_path = os.path.join(work_dir, 'redline_error.png')
    line_img = cv2.imread(temp_path)  # BGR
    h, w = line_img.shape[:2]
    cv2.imshow('ori_img', line_img)

    # 先不使用高斯，及边缘检测
    # line_img = cv2.GaussianBlur(line_img, (5, 5), 0)   # 高斯模糊然后在二值化就不太好了
    # cv2.imshow('blur_img', line_img)

    # plt.hist(line_img.ravel(), 256)
    # plt.show()


    # 1 计算与背景的距离，根据阈值分割，背景bgr值如何定义
    # distance_img = _distance_bg(line_img, [150,150,150], 6000)
    # save_name = os.path.join(work_dir,'distance.png')
    # cv2.imwrite(save_name,distance_img)
    # cv2.imshow('distance_img', distance_img)

    # 2 是否先分离背景更好,将背景变白
    # line_img = _distance_bg(line_img, [150, 150, 150], 6000)
    # line_img = cv2.resize(line_img, (w, h*3))

    # 2 分离红色通道
    red_ch = line_img[:,:,1] #BGR
    cv2.imshow('red chnnel',red_ch)


    # 3 灰度图类间方差最大取阈值
    gray_line = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    thre = _var_max_interclass(gray_line)
    print('maxvar threshold:', thre)
    #写了半天就是人家一个参数
    # _, gray_line = cv2.threshold(gray_line, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, gray_line = cv2.threshold(gray_line,thre,255,cv2.THRESH_BINARY)
    cv2.imshow('gray_line',gray_line)


    # save_name = os.path.join(work_dir,'maxvar-blur.png')
    # cv2.imwrite(save_name, gray_line)





    cv2.waitKey(0)
    cv2.destroyAllWindows()