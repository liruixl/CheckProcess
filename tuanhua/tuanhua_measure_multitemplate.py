import cv2
import os
from enum import Enum
import numpy as np
from collections import Counter
from tools.imutils import cv_imread, separate_color
import tools.check as check

"""
团花大小约:(369*145) 369部分左侧不够
以右上角为参考：团花位置距离右侧25，距离上10
以此可大致确定团花位置
可作为常量使用
"""
TH_W = 384  # 374
TH_H = 150  # 145
# TH_XOFF_RIGHT = 25  # hualing票据中靠右，团花
TH_XOFF_RIGHT = 20

TH_YOFF_TOP = 5   # hualing中有更靠上的


class ThType(Enum):
    TH_UNKNOW = 0
    TH_BRIGHT = 1
    TH_NORMAL = 2
    TH_LIGHT = 3


def paste_to_canvas(dst_h, dst_w, y_off, x_off, img):
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)  # 扩大20像素
    h, w = img.shape
    dst[y_off:y_off+h, x_off:x_off+w] = img
    return dst

class TmpRegionInfo:
    def __init__(self, xmin, ymin, xmax, ymax, oriimg):
        """
        :param xmin: 记录小模板自身在原图的坐标区域
        :param ymin:
        :param xmax:
        :param ymax:
        :param oriimg: 原图
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.tmp_img = oriimg[ymin:ymax, xmin:xmax]


class TuanHuaROI:
    def __init__(self, checkdir:check.CheckDir, bbox=None):
        """

        :param upuv_img: 紫外反射图像, 检测黑色涂改墨团
        :param upirtr_img: 红外透视图像, 检测刮擦
        :param upg_img: 可见光图像, 得到文字掩膜
        :param bbox: 团花坐标，若None，使用先验坐标
        """

        upg_img = checkdir.upg
        upir_img = checkdir.upir
        upirtr_img = checkdir.upirtr
        upuv_img = checkdir.upuv

        check_h, check_w, check_c = upuv_img.shape

        # 预定义位置
        xmin = check_w - TH_W - TH_XOFF_RIGHT
        ymin = TH_YOFF_TOP
        xmax = xmin + TH_W
        ymax = ymin + TH_H

        if bbox is not None:
            assert len(bbox) == 4
            xmin, ymin, xmax, ymax = bbox
            assert check_w >= xmax > xmin >= 0 and check_h > ymax > ymin >= 0

        self.upg_box = upg_img[ymin:ymax, xmin:xmax]
        self.upir_box = upir_img[ymin:ymax, xmin:xmax]
        self.upirtr_box = upirtr_img[ymin:ymax, xmin:xmax]
        self.upuv_box = upuv_img[ymin:ymax, xmin:xmax]

        # 在紫外下要忽略文字掩膜区域
        self.text_mask, _ = separate_color(self.upg_box, color="black")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.text_mask = cv2.dilate(self.text_mask, kernel, iterations=1)

        # 团花的分类
        self.th_type = self._check_tuanhua_type(self.upuv_box)

        self.tuanhua_bin = None
        if self.th_type == ThType.TH_BRIGHT:
            self.tuanhua_bin, _ = separate_color(self.upuv_box, color='th_bright')
        else:
            self.tuanhua_bin, _ = separate_color(self.upuv_box, color='th_normal')

    def _check_tuanhua_type(self, tuanhua_img):
        """

        :param tuanhua_img: 紫外下团花裁剪图像
        :return: 团花类型
        类型  pix_sum  v_mean
        淡化： 9130 180
        正常：20000+ 205
        偏黄：22000，219
        发亮：24000，220
        """
        tuanhua_hsv = cv2.cvtColor(tuanhua_img, cv2.COLOR_BGR2HSV)

        # 首先以较大阈值提取出团花图案
        th_mask, th_bgr = separate_color(self.upuv_box, color='th_normal')

        # 根据【像素数+亮度】两个维度，判断团花类型
        self.pixel_sum = np.sum(th_mask > 10)

        th_v = tuanhua_hsv[:, :, 2]  # (H*W*1)

        v_sum = 0
        h, w = th_mask.shape[:2]
        for y in range(0, h):
            for x in range(0, w):
                if th_mask[y][x] > 0:
                    v_sum += th_v[y][x]

        self.v_mean = v_sum // self.pixel_sum


        # print('tuanhua pixel sum:', pixel_sum, end=' ')
        # print('tuanhua V mean:', self.v_mean)
        #
        # cv2.imshow('hsv msk', th_mask)
        # cv2.imshow('TH', tuanhua_img)
        # cv2.waitKey(0)

        if self.pixel_sum < 11000 and self.v_mean < 205:
            return ThType.TH_LIGHT

        if self.v_mean >= 222 and self.pixel_sum > 22000:
            return ThType.TH_BRIGHT

        return ThType.TH_NORMAL

        # 之前的规则，使用全图的v_mean, 不太合适
        # if self.v_mean > 135:
        #     return ThType.TH_BRIGHT
        # elif self.v_mean > 110:
        #     return ThType.TH_NORMAL
        # elif self.v_mean > 80:
        #     return ThType.TH_LIGHT
        #
        # return ThType.TH_UNKNOW


class TuanhuaMeasure:
    def __init__(self):
        self.debug = False
        self.template_normal_list = []  # type:list[TuanHuaROI]
        self.template_light_list = []   # type:list[TuanHuaROI]


    def add_tuanhua_template(self, checkdir_path, bbox=None):

        assert os.path.isdir(checkdir_path)

        check_dir = check.CheckDir()
        check_dir.init_by_checkdir(checkdir_path)

        th_roi = TuanHuaROI(check_dir)

        if th_roi.th_type == ThType.TH_LIGHT:
            self.template_light_list.append(th_roi)
        else:
            self.template_normal_list.append(th_roi)

        print('you add a tuanhua template:', th_roi.th_type)


    def measure(self, check_dir:check.CheckDir, bbox=None):
        th_dect = TuanHuaROI(check_dir, bbox)

        # 先去红外图检测刮擦或者墨团。


        # 然后紫外下检测刮擦
        score = 1.0

        print('the detect tuanhua type:', th_dect.th_type, th_dect.pixel_sum,th_dect.v_mean)

        if th_dect.th_type == ThType.TH_LIGHT:
            score = self._measure_light(th_dect)
        else:
            score = self._measure_normal(th_dect)

        return score

    def _measure_normal(self, tuanhua_roi:TuanHuaROI):

        score_list = []

        for i in range(len(self.template_normal_list)):

            if self.debug:
                print('start to match id:{0} tuanhua template... '.format(i), end=' ')

            th_temp = self.template_normal_list[i]
            conf = self._measure_with_one_template(tuanhua_roi, th_temp)
            score_list.append(conf)
            # if conf > 0.5:
            #     break

            if self.debug:
                print('conf score: {1},'.format(i, conf))

        if max(score_list) < 0.5:
            print('#############,score list:', score_list)

        return max(score_list)

    def _measure_light(self, tuanhua_roi:TuanHuaROI):

        return 1.0

    def _measure_with_one_template(self, dectroi:TuanHuaROI,
                                   temproi:TuanHuaROI):

        # 左右小模板去匹配，分别考察左右部分区域
        l_temp_img = TmpRegionInfo(2, 58 - 20, 30 + 30, 90 + 20, temproi.upuv_box)
        r_temp_img = TmpRegionInfo(325, 74-50, 325+50, 74+50, temproi.upuv_box)


        H, W = temproi.upuv_box.shape[:2]
        det_H, det_W = dectroi.upuv_box.shape[:2]

        padding = 10

        # 四周扩大padding像素b 并将模板放在画布中间
        # canvas = np.zeros((H + padding * 2, W + padding * 2), dtype=np.uint8)
        # canvas[padding: H + padding, padding:W + padding] = temproi.tuanhua_bin

        # 抽象为一个函数
        canvas = paste_to_canvas(H+padding*2, W+padding*2,
                                 padding, padding, temproi.tuanhua_bin)
        text_mask_1 = cv2.bitwise_not(temproi.text_mask)
        text_mask_1 = paste_to_canvas(H + padding * 2, W + padding * 2,
                                      padding, padding, text_mask_1)

        def to_aligned_by(tmp_region):

            result = cv2.matchTemplate(dectroi.upuv_box, tmp_region.tmp_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            t_left = max_loc  # (x,y) 访问 result[y][x]
            det_xmin, det_ymin = t_left

            dx = det_xmin - tmp_region.xmin
            dy = det_ymin - tmp_region.ymin

            if abs(dx) > padding or abs(dy) > padding:
                return None

            det_x = padding-dx
            det_y = padding-dy

            det_canvas = paste_to_canvas(H+padding*2, W+padding*2,
                                         det_y, det_x, dectroi.tuanhua_bin)

            """
            现在，canvas 与 det_canvas 上的团花图案对齐了
            """

            xor_mask = cv2.bitwise_xor(canvas, det_canvas)

            # 不考虑文字区域

            text_mask_2 = cv2.bitwise_not(dectroi.text_mask)
            text_mask_2 = paste_to_canvas(H+padding*2, W+padding*2, det_y, det_x, text_mask_2)

            text_mask_all = cv2.bitwise_and(text_mask_1, text_mask_2)

            xor_mask_not_text = cv2.bitwise_and(xor_mask, text_mask_all)

            return xor_mask_not_text

        xor_1 = to_aligned_by(l_temp_img)
        xor_2 = to_aligned_by(r_temp_img)

        # 模板匹配位置偏差太大
        if xor_1 is None or xor_2 is None:
            return 0.0

        # 分割成块，该选择哪一个xor, 通过白色像素数判断
        canvas_h, canvas_w = xor_1.shape[:2]

        # x_list = [9, 6, 7, 6, 9] # 实验得到团花如何分割x轴方向，所占比例
        x_list = [10, 6, 7, 6, 10]  # 扩展了padding，应该是10
        times = sum(x_list)
        sw = canvas_w // times

        w_range_list = []
        l_range_1 = [0, sw*10]
        r_range_1 = [canvas_w-sw*10, canvas_w]
        l_range_2 = [l_range_1[1], l_range_1[1]+sw*6]
        r_range_2 = [r_range_1[0] - sw*6, r_range_1[0]]

        # 中间区域由剩余部分确认
        mid_range = [l_range_2[1], r_range_2[0]]

        w_range_list.append(l_range_1)
        w_range_list.append(l_range_2)
        w_range_list.append(mid_range)
        w_range_list.append(r_range_2)
        w_range_list.append(r_range_1)

        # print('tuanhua split areas:',w_range_list)

        xor_res = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        for x1, x2 in w_range_list:
            for y1, y2 in [[0, canvas_h//2], [canvas_h//2, canvas_h]]:
                w_sum_1 = np.sum(xor_1[y1:y2, x1:x2] > 1)
                w_sum_2 = np.sum(xor_2[y1:y2, x1:x2] > 1)

                if w_sum_1 > w_sum_2:
                    xor_res[y1:y2, x1:x2] = xor_2[y1:y2, x1:x2]
                else:
                    xor_res[y1:y2, x1:x2] = xor_1[y1:y2, x1:x2]

        if self.debug:

            # cv2.imshow('region_l', l_temp_img.tmp_img)
            # cv2.imshow('region_r', r_temp_img.tmp_img)

            cv2.imshow('tmp and dect', np.hstack((temproi.upuv_box,
                                                  dectroi.upuv_box)))

            cv2.imshow('tmp and dect bin', np.hstack((temproi.tuanhua_bin,
                                                 dectroi.tuanhua_bin)))

            cv2.imshow('xor_res', np.hstack((xor_1, xor_2, xor_res)))


        conf = self._mask2confandres(xor_res)


        return conf


    def _mask2confandres(self, mask):

        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ero = cv2.erode(mask, tempkernel, iterations=2)

        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ero = cv2.dilate(ero, tempkernel, iterations=2)

        ero_show = cv2.cvtColor(ero, cv2.COLOR_GRAY2BGR)

        _, contours, hierarchy = cv2.findContours(ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bianzaolist = []

        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            bianzaolist.append(area)

            if area > 10:
                cv2.rectangle(ero_show, (x, y), (x + w, y + h), (0, 0, 200))

        bianzaolist.sort(reverse=True)


        if self.debug:
            cv2.imshow('ero img', ero_show)
            cv2.waitKey(0)

        conf = 1.0

        # 太多了感觉是误检
        if len(bianzaolist) >= 15 and bianzaolist[0] < 30:
            return 0.88

        for ar in bianzaolist:
            if ar > 50:
                return 0.1
            if ar > 40:
                return 0.2
            if ar > 20:
                return 0.3
            if ar > 10:
                return 0.4

        # if 1 < len(bianzaolist) < 5:
        #     conf = 0.9

        return conf



if __name__ == '__main__':

    th_measure = TuanhuaMeasure()

    path_1 = r'../tuanhua/20200828_14-44-40_v147'
    path_2 = r'../tuanhua/20200828_14-45-08_v128'
    path_3 = r'../tuanhua/20200828_15-39-29'

    th_measure.add_tuanhua_template(path_1)
    th_measure.add_tuanhua_template(path_2)
    th_measure.add_tuanhua_template(path_3)

    # tuanhuayc = r'E:\异常图0927\敦南\敦南正常'
    # tuanhuayc = r'E:\异常图0927\敦南\敦南异常'

    tuanhuayc = r'E:\异常图0927\华菱\正常票'

    # tuanhuayc = r'E:\异常图0927\敦南正常细分\团花\团花黄'

    names = os.listdir(tuanhuayc)

    names = [n for n in names if '20200904_02-23-33' in n]
    th_measure.debug = True

    for na in names:
        print('===============processing ', na, '=======================')
        imgdir = os.path.join(tuanhuayc, na)

        cd = check.CheckDir()
        cd.init_by_checkdir(imgdir)

        conf = th_measure.measure(cd, bbox=None)

        # 测试团花类型判断
        # th_roi = TuanHuaROI(cd)



