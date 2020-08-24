import numpy as np
import cv2
from measure.stitcher import Stitcher


class BankMeasure:
    def __init__(self, icon_path):

        self.debug = True
        self.icon_temp = cv2.imread(icon_path)

        # _, blur_bin = cv2.threshold(blur_avg, thre, 255, cv2.THRESH_BINARY)
        # _, blur_bin = cv2.threshold(blur_avg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        icon_gray = cv2.cvtColor(self.icon_temp, cv2.COLOR_BGR2GRAY)
        _, self.icon_bin = cv2.threshold(icon_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.stitcher = Stitcher()

    def measure(self, img):
        off_or_none = self.stitcher.stitch(img, self.icon_temp)

        if off_or_none is None:
            return 0.0

        off_x, off_y = off_or_none
        assert off_x > 0 and off_y > 0

        H, W = self.icon_temp.shape[:2]

        aligned = img[off_y : off_y + H, off_x : off_x + W]
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 模板icon_bin与aligned对齐，且大小一致了
        # 应该全为0，黑色就对了
        # 但实际上会有噪点 再做一次腐蚀膨胀
        mask = cv2.bitwise_xor(self.icon_bin, binary)
        conf = self._convert_mask2conf(mask)

        print('this dectect bank icon confidence is:', conf)
        return conf

    def _convert_mask2conf(self, mask):
        """
        :param mask: binary img
        :return: conf
        """
        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ero = cv2.erode(mask, tempkernel, iterations=2)
        res = cv2.dilate(ero, tempkernel, iterations=2)
        if self.debug:
            cv2.imshow('mask', np.hstack((mask, res)))
            cv2.waitKey(0)

        _, contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        areas = []
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            areas.append(area)
            print('area & rect:',area, w*h)

        areas.sort(reverse=True)

        conf = 1.0

        if len(areas) > 10:
            return 0.1

        for oo in areas:
            if oo > 300:
                return 0.1
            if oo > 100:
                return 0.2
            if oo > 50:
                return 0.3

            conf -= 0.05

        return conf

    def measure_with_loc(self, upuv_path, bnd_box):
        img = cv2.imread(upuv_path)
        xmin, ymin, xmax, ymax = bnd_box
        img = img[ymin : ymax, xmin, xmax]
        return self.measure(img)


if __name__ == '__main__':
    bank_temp = r'E:\DataSet\measure_dataset\template\bank_icon\zhongnong_uv.jpg'
    bank_measure = BankMeasure(bank_temp)
    bank_measure.debug = True

    dect = r'../hanghui/img_dect/zhongnong_1.jpg'
    img = cv2.imread(dect)
    cv2.imshow('dect', img)

    bank_measure.measure(img)