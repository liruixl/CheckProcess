import cv2
import numpy as np

'''
1 实验得到HSV范围、纤维丝长度、宽度等范围
2 度量Wrapper，开放接口
'''


class FiberMeasure:

    def __init__(self):
        self.debug = False
        # 深蓝：【100，70，130】 【120，190，255】
        # 浅蓝：【100，70，75】 【150，190，255】
        # 红：【130/135，60，65】【160， 150/160，140】
        self.blue_l_hsv1 = [100, 70, 130]
        self.blue_h_hsv1 = [120, 190, 255]

        self.blue_l_hsv2 = [100, 70, 110]  #(100, 70, 75/110)
        self.blue_h_hsv2 = [150, 190, 255]

        self.blue_l_hsv3 = [100, 70, 75]
        self.blue_h_hsv3 = [150, 190, 255]

        self.red_l_hsv = [130, 60, 65]
        self.red_h_hsv = [160, 150, 140]

        self.hsv_ranges = []
        self.hsv_ranges.append((self.blue_l_hsv1, self.blue_h_hsv1))
        self.hsv_ranges.append((self.blue_l_hsv2, self.blue_h_hsv2))
        self.hsv_ranges.append((self.blue_l_hsv3, self.blue_h_hsv3))

        self.hsv_ranges.append((self.red_l_hsv, self.red_h_hsv))

        self.length = (5, 70, 80)  # (min, normal, max)
        self.width = (0.0, 3.5, 4.5)

    def measure(self, img):
        minl, normal, maxl = self.length
        minw, nw, maxw = self.width

        def get_length_conf(l):
            if minl <= l <= maxl:
                if l <= normal:
                    return 1.0
                else:
                    return 0.9

            if 0 <= l <= minl:
                return 0.75
            if l >= maxl:
                return 0.3

            return 0.2

        def get_width_conf(w):
            if minw <= w <= maxw:
                if w <= nw:
                    return 1.0
                else:
                    return 0.8
            if w > maxw:
                return 0.4

            return 0.2

        confs = []

        for low_hsv, high_hsv in self.hsv_ranges:
            mask = self.separate_color(img, low_hsv, high_hsv)
            tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask = cv2.erode(mask, tempkernel, iterations=1)
            mask = cv2.dilate(mask, tempkernel, iterations=1)
            h, w = mask.shape

            # 防止轮廓出界，以更大的画布做背景
            paint = np.zeros((h + 6, w + 6), dtype=np.uint8)
            paint[3: 3 + h, 3: 3 + w] = mask
            mask = paint

            length = self.fiber_length(mask)
            width = self.fiber_width(mask)

            score_l = get_length_conf(length)
            score_w = get_width_conf(width)

            c = (score_l + score_w) / 2
            confs.append(c)

            print('L {}得分:{}, W {}得分:{}, 平均{}'.format(length,score_l, width, score_w, c))

        print('得分列表', confs)
        return max(confs)


    def fiber_length(self, gray):
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 10 or len(contours) < 1:
            # print("find {0} > 1 contours".format(len(contours)))
            return -1

        fiber_lenght = 0

        for cont in contours:
            area = cv2.contourArea(cont)
            if area < 10:
                print('L ignore suspected area:', area)
                continue
            fiber_lenght += cv2.arcLength(cont, True)

        if self.debug:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.imshow('length', debug_img)
            cv2.waitKey(0)

        return fiber_lenght//2


    def fiber_width(self, gray):
        H, W = gray.shape
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)

        n = len(contours)

        if n == 0:
            return -1

        contours = contours[: min(5, n)]
        # areas = [cv2.contourArea(cont) for cont in contours ]

        maxdist = 0

        for cont in contours:
            area = cv2.contourArea(cont)
            if area < 10:  # 从大到小排列，这里可以退出了
                break
            center = (0, 0)
            for i in range(0, H):
                for j in range(0, W):
                    dist = cv2.pointPolygonTest(cont, (i, j), True)
                    if dist > maxdist:
                        maxdist = dist
                        center = (i, j)

        return int(maxdist*1000)/1000

    def separate_color(self, img, l_hsv, h_hsv):
        """

        :param img: BGR
        :param l_hsv:
        :param h_hsv:
        :return: mask
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_hsv = np.array(l_hsv)
        high_hsv = np.array(h_hsv)

        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)

        return mask


if __name__ == '__main__':
    import os
    blue_fiber_dir = r'E:\DataSet\fiber\f_red'
    names = os.listdir(blue_fiber_dir)
    paths = [os.path.join(blue_fiber_dir, n) for n in names]

    fm = FiberMeasure()
    fm.debug = True

    paths = [r'E:\DataSet\fiber\f_blue\00009.jpg']

    for ppp in paths:
        img = cv2.imread(ppp)
        fm.measure(img)

