"""
支票相关类
"""

import os
import numpy as np
import cv2
from tools.imutils import cv_imread, separate_color

class CheckDir:
    def __init__(self):

        # 常用
        self.upg = None
        self.upir = None
        self.upirtr = None
        self.upuv = None

        # 不常用
        self.upuvtr = None
        self.dwg = None
        self.dwir = None
        self.dwirtr = None
        self.dwuv = None
        self.dwuvtr = None

    def init_by_checkdir(self, checkdir_path, ext='.bmp'):
        self.upg    = cv_imread(os.path.join(checkdir_path, 'UpG' + ext))
        self.upir   = cv_imread(os.path.join(checkdir_path, 'Upir' + ext))
        self.upirtr = cv_imread(os.path.join(checkdir_path, 'Upirtr' + ext))
        self.upuv   = cv_imread(os.path.join(checkdir_path, 'Upuv' + ext))
        self.upuvtr = cv_imread(os.path.join(checkdir_path, 'Upuvtr' + ext))
        self.dwg    = cv_imread(os.path.join(checkdir_path, 'DwG' + ext))
        self.dwir   = cv_imread(os.path.join(checkdir_path, 'Dwir' + ext))
        self.dwirtr = cv_imread(os.path.join(checkdir_path, 'Dwirtr' + ext))
        self.dwuv   = cv_imread(os.path.join(checkdir_path, 'Dwuv' + ext))
        self.dwuvtr = cv_imread(os.path.join(checkdir_path, 'Dwuvtr' + ext))

    def set_upg(self, img):
        self.upg = img

    def set_upir(self, img):
        self.upir = img

    def set_upirtr(self, img):
        self.upirtr = img

    def set_upuv(self, img):
        self.upuv = img



def to_detect_scratch(checkdir:CheckDir, bbox):

    # 文字区域梯度肯定大，要排除这部分
    xmin, ymin, xmax, ymax = bbox
    upg_bbox = checkdir.upg[ymin:ymax, xmin:xmax]
    upirtr_bbox = checkdir.upirtr[ymin:ymax, xmin:xmax]

    text_mask, _ = separate_color(upg_bbox, color="black")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)

    # 红外反射下
    upirtr_gray = cv2.cvtColor(upirtr_bbox, cv2.COLOR_BGR2GRAY)
    # minValue取值较小，边界点检测较多,maxValue越大，标准越高，检测点越少
    v1 = cv2.Canny(upirtr_gray, 150, 300)  # (80, 250)  (150, 300)
    v1_not_text = cv2.bitwise_and(v1, cv2.bitwise_not(text_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    v1_not_text = cv2.dilate(v1_not_text, kernel, iterations=1)
    v1_not_text = cv2.erode(v1_not_text, kernel, iterations=1)

    v1_show = cv2.cvtColor(v1_not_text, cv2.COLOR_GRAY2BGR)

    # 亮度图
    upirtr_gray[upirtr_gray <= 245] = 0
    big245_show = cv2.cvtColor(upirtr_gray, cv2.COLOR_GRAY2BGR)

    _, contours, hierarchy = cv2.findContours(v1_not_text, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

    suspect_rects = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 64:
            suspect_rects.append([x, y, w, h])
            cv2.rectangle(v1_show, (x, y), (x + w, y + h), (0, 0, 250))

    scratch_bbox = []

    for bbox in suspect_rects:
        x, y, w, h = bbox
        crop = upirtr_gray[y:y+h, x:x+w]
        s = np.sum(crop >= 240)

        # print('ssss', s, w*h)
        if 30*s > w*h:
            cv2.rectangle(big245_show, (x, y), (x + w, y + h), (0, 0, 250))
            scratch_bbox.append([x, y, x+w, y+h])



    cv2.imshow('tuanhua', np.hstack([upg_bbox, upirtr_bbox, v1_show, big245_show]))
    cv2.imshow('canny', np.hstack([text_mask, v1, v1_not_text, upirtr_gray]))
    cv2.waitKey(0)


    return scratch_bbox


if __name__ == '__main__':

    from tuanhua.tuanhua_measure_multitemplate import get_th_bbox

    dir_path = r'E:\异常图0927\郭南异常细分\团花异常'
    names = os.listdir(dir_path)


    for na in names:
        print('===============processing ', na, '=======================')
        imgdir = os.path.join(dir_path, na)

        cd = CheckDir()
        cd.init_by_checkdir(imgdir)

        bbox = get_th_bbox(cd.upg.shape[1])
        res = to_detect_scratch(cd, bbox)

