
import cv2
import numpy as np
from collections import Counter


class TuanhuaMeasure:

    def __init__(self, bloom_path):
        self.debug = False

        im = cv2.imread(bloom_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # _, gray = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 只寻找最外轮廓 RETR_EXTERNAL RETR_CCOMP
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        H, W = gray.shape

        self.roi_mask = np.zeros((H, W), dtype=np.uint8)

        # 绘制ROI mask
        # cv2.drawContours(im, contours, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > H * W // 2:
                count += 1
                cv2.drawContours(self.roi_mask, [cnt], 0, 255, -1, lineType=cv2.LINE_AA)

        assert count == 1, 'Non-standard group flower image template'

        # 以右上角为原点，左x，下y，这个区域是文字区域
        self.ignore_x_range = (110, 310)
        self.ignore_y_range = (80, 160)

        # 以团花为参考系，文字区域和收款行区域
        self.num_box = [100, 70, 290, 120]
        self.recvbank_box = (0, 100)  # (xmin, ymin)

        self.area_thresh = 20

    def measure(self, dect_img, upir_img_path, bnd_box):
        """

        :param dect_img: dectected bloom box img
        :param upir_img_path: upir check img
        :param bnd_box: [xmin, ymin, xmax, ymax]
        :return: confidence
        """

        upir_img = cv2.imread(upir_img_path)
        xmin, ymin, xmax, ymax = bnd_box
        upir_box = upir_img[ymin:ymax, xmin:xmax]

        # 利用红外反射图像上检测是否有涂改（黑），刮擦（白）
        upir_tuanhua = cv2.imread(upir_img_path)
        blank_list = self.dect_blank_areas(upir_tuanhua)
        blank_list.sort(reverse=True)

        white_list = self.dect_wihte_area(upir_tuanhua)

        return 1.0

    def dect_blank_areas(self, upir_tuanhua):
        """

        :param upir_tuanhua: 红外反射上的团花图像
        :return: 检测到黑墨团的面积列表
        """
        gray = cv2.cvtColor(upir_tuanhua, cv2.COLOR_BGR2GRAY)

        H, W = gray.shape
        num_mask = np.zeros((H, W), dtype=np.uint8)  # 流水号
        bankname_mask = np.zeros((H, W), dtype=np.uint8)  # 收款行

        xmin, ymin, xmax, ymax = self.num_box
        num_mask[ymin: ymax, xmin: xmax] = 255
        bankname_mask[self.recvbank_box[1]:, self.recvbank_box[0]:] = 255
        mask = cv2.bitwise_or(num_mask, bankname_mask)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_or(binary, mask)

        binary = cv2.bitwise_not(binary)  # 变造区域呈现白色255, 求轮廓是求白色的轮廓，否则是最外层

        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            print('area:', area)
            if area > self.area_thresh:
                areas.append(area)
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        if self.debug:
            cv2.imshow('mask', np.hstack((num_mask, bankname_mask)))
            cv2.imshow('binary', np.hstack((gray, binary)))
            cv2.imshow('contours', img)
            cv2.waitKey(0)

        return areas


    def dect_wihte_area(self, upir_tuanhua):
        """
        把所有黑色除掉，剩余灰度图，再次均方差二值化THRESH_OTSU
        如果是刮擦，那么白上加白
        :param upir_tuanhua:
        :return:
        """
        gray = cv2.cvtColor(upir_tuanhua, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.erode(binary, tempkernel, iterations=1)

        gray = cv2.bitwise_and(gray, binary)
        thre = _var_max_interclass(gray, calc_zero=False)
        print(thre)
        # cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('adapt', th2)


        cv2.imshow('gray', gray)
        cv2.imshow('binary', binary)
        cv2.waitKey(0)

        return []


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
    bloom_path = r'../hanghui/img_tuanhua/tuanhua_uv_4.jpg'  # 这个最好了
    # upir_tuanhua = r'../hanghui/img_tuanhua/test_ir.jpg'
    upir_tuanhua = r'../hanghui/img_tuanhua/test_ir_bai.jpg'
    # upir_tuanhua = r'gray_white.jpg'


    img = cv2.imread(upir_tuanhua)
    tm = TuanhuaMeasure(bloom_path)
    tm.debug = True

    # blank_list = tm.dect_blank_areas(img)
    # print(blank_list)
    white_list = tm.dect_wihte_area(img)
