import numpy as np
import cv2
import os
from tools.imutils import cv_imread, separate_color
from measure.stitcher import Stitcher

"""
行徽坐标信息： 
中心点(1140, 78) (1137, 74)
小农行行徽大小：88*88
大农行行徽大小：96*96
"""

BANK_W = 114
BANK_H = 114

# BANK_XOFF_RIGHT = 1140 - BANK_W//2
BANK_XOFF_RIGHT = 1135 - BANK_W//2

BANK_YOFF_TOP = 70 - BANK_H//2


def paste_to_canvas(dst_h, dst_w, y_off, x_off, img):
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)
    h, w = img.shape
    dst[y_off:y_off+h, x_off:x_off+w] = img
    return dst


class BankUpuv:
    def __init__(self, bank_upuv):
        self.icon_bgr = bank_upuv.copy()

        # 直接二值化导致纤维丝也会被当作行徽
        # self.icon_gray = cv2.cvtColor(self.icon_bgr, cv2.COLOR_BGR2GRAY)
        # _, self.icon_bin = cv2.threshold(self.icon_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 提取黄色
        det_hsv_mask, det_hsv = separate_color(bank_upuv, color='bank_yellow')

        self.icon_gray = cv2.cvtColor(det_hsv, cv2.COLOR_BGR2GRAY)
        _, self.icon_bin = cv2.threshold(self.icon_gray , 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



class BankMeasure:
    def __init__(self):
        self.debug = False
        self.template_list_upuv = []  # type: list[BankUpuv]
        self.stitcher = Stitcher(debug=self.debug)

        self.reasons = []  # type:list[str]

    def add_template_of_upuv(self, upuv_icon_path):
        img = cv_imread(upuv_icon_path)
        assert img is not None
        self.template_list_upuv.append(BankUpuv(img))

    def measure(self, upuv_img, bank_icon_bbox=None):

        self.reasons.clear()

        if len(self.template_list_upuv) < 1:
            raise RuntimeError('not template to match!')

        h, w, c = upuv_img.shape

        xmin = w - BANK_W - BANK_XOFF_RIGHT
        ymin = BANK_YOFF_TOP
        xmax = xmin + BANK_W
        ymax = ymin + BANK_H

        if bank_icon_bbox is not None:
            assert len(bank_icon_bbox) == 4
            xmin, ymin, xmax, ymax = bank_icon_bbox

        img = upuv_img[ymin:ymax, xmin:xmax]

        if self.debug:
            cv2.imshow('dect bank', img)
            cv2.waitKey(0)

        bank_upuv_dect = BankUpuv(img)
        conf_list = []

        for i in range(len(self.template_list_upuv)):
            bank_upuv_temp = self.template_list_upuv[i]
            score = self.measure_with_one_template(bank_upuv_dect, bank_upuv_temp)
            conf_list.append(score)

            if self.debug:
                print('BankMeasure::measure(),第{0}个模板得分{1}'.format(i, score))

        return max(conf_list)

    def measure_with_one_template(self, bank_dect:BankUpuv,
                                  bank_template:BankUpuv)->float:

        # bgr图像做特征点匹配,
        # 并得到检测图特征点坐标相对于模板特征点坐标的偏移
        off_or_none = self.stitcher.stitch(bank_dect.icon_bgr,
                                           bank_template.icon_bgr,
                                           showMathes=self.debug)

        if off_or_none is None:
            self.reasons.append('not match any feature point.')
            return 0.0

        off_x, off_y = off_or_none

        if abs(off_x) > 30 or abs(off_y) > 30:
            self.reasons.append('maybe position error.')
            return 0.1

        h_dect, w_dect = bank_dect.icon_bgr.shape[:2]
        h_temp, w_temp = bank_template.icon_bgr.shape[:2]

        wh = max(h_dect, w_dect, h_temp, w_temp) + 60
        assert wh < 200

        icon_move = paste_to_canvas(wh, wh, 30, 30, bank_template.icon_bin)
        aligned = paste_to_canvas(wh, wh, 30-off_y, 30-off_x,
                                  bank_dect.icon_bin)

        move9s = []
        ioff = [-1, 0, 1]
        joff = [-1, 0, 1]

        for i in ioff:
            for j in joff:
                M = np.float32([[1, 0, i], [0, 1, j]])
                dst = cv2.warpAffine(icon_move, M, (wh, wh))
                move9s.append(dst)

        def get255cnt(img):
            cnt = 0
            h, w = img.shape
            for i in range(h):
                for j in range(w):
                    if img[i][j] > 2:
                        cnt += 1
            return cnt

        masks = [cv2.bitwise_xor(move_icon, aligned) for move_icon in move9s]
        masks_255_cnt = [get255cnt(m) for m in masks]
        minidx = masks_255_cnt.index(min(masks_255_cnt))

        icon_move = move9s[minidx]  # 最优的模板偏移位置
        xor_img = cv2.bitwise_xor(icon_move, aligned)

        if self.debug:
            cv2.imshow('move9', np.hstack(masks))
            cv2.imshow('xor', np.hstack((icon_move, aligned)))
            cv2.imshow('xor_mask', xor_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        conf = self._convert_mask2conf(xor_img)

        return conf

    def _convert_mask2conf(self, mask):
        """
        :param mask: xor binary img
        :return: conf
        """
        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ero = cv2.erode(mask, tempkernel, iterations=1)
        res = cv2.dilate(ero, tempkernel, iterations=2)

        _, contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        areas = []
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            areas.append(area)

        areas.sort(reverse=True)

        if self.debug:
            cv2.imshow('final ero', res)
            print('bank bad area:', areas)
            cv2.waitKey(0)

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
            if oo > 18:
                return 0.4
            conf -= 0.11

        return conf


if __name__ == '__main__':
    bank_measure = BankMeasure()

    nonghang_1 = r'../bank_icons/01nonghang.jpg'
    nonghang_2 = r'../bank_icons/20nonghang2.jpg'
    nonghang_3 = r'../bank_icons/21nonghang3.jpg'
    nonghang_4 = r'../bank_icons/22nonghang4.jpg'

    bank_measure.add_template_of_upuv(nonghang_1)
    bank_measure.add_template_of_upuv(nonghang_2)
    bank_measure.add_template_of_upuv(nonghang_3)
    bank_measure.add_template_of_upuv(nonghang_4)


    # test_check_dir = r'E:\异常图0927\敦南\敦南异常'
    test_check_dir = r'E:\异常图0927\敦南\敦南正常'
    # test_check_dir = r'E:\异常图0927\华菱\正常票'

    check_ids = os.listdir(test_check_dir)
    # check_ids = [id for id in check_ids if '行徽' in id]
    check_ids = [id for id in check_ids if '15-41-04' in id]

    bank_measure.debug = True

    for na in check_ids:
        print('===========',na,'=================================')
        imgdir = os.path.join(test_check_dir, na)
        upuv = imgdir + '/' + 'Upuv.bmp'
        upg = imgdir + '/' + 'UpG.bmp'
        upir = imgdir + '/' + 'Upir.bmp'
        upirtr = imgdir + '/' + 'Upirtr.bmp'

        upuv_img = cv_imread(upuv)


        score = bank_measure.measure(upuv_img)
        if score < 0.5:
            print(na, '得分：', score)
            print('---reason--->', bank_measure.reasons)