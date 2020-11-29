
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

        self.blank_thresh = 20
        self.white_thresh = 3

    def measure(self, dect_img, upirtr_img_path, bnd_box):
        """

        :param dect_img: dectected bloom box img
        :param upirtr_img_path: upir check img
        :param bnd_box: [xmin, ymin, xmax, ymax]
        :return: confidence
        """

        upir_img = cv2.imread(upirtr_img_path)
        xmin, ymin, xmax, ymax = bnd_box
        upir_box = upir_img[ymin:ymax, xmin:xmax]

        # 利用红外反射图像上检测是否有涂改（黑），刮擦（白）
        # upirtr_tuanhua = cv2.imread(upirtr_img_path)
        blank_list = self.dect_blank_areas(upir_box)
        blank_list.sort(reverse=True)

        temp = sum(blank_list)
        
        if temp > 150:
            return 0.1
        if temp > 100:
            return 0.2
        if temp > 50:
            return 0.3

        white_list = self.dect_wihte_area(upir_box)
        white_list.sort(reverse=True)

        temp = sum(white_list)

        if temp > 100:
            return 0.1
        if temp > 50:
            return 0.2
        if temp > 30:
            return 0.3

        if len(blank_list) > 2 or len(white_list) > 5:
            return 0.85

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
            if area > self.blank_thresh:
                areas.append(area)
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        if self.debug:
            cv2.imshow('mask', np.hstack((num_mask, bankname_mask)))
            cv2.imshow('binary', np.hstack((gray, binary)))
            cv2.imshow('contours', img)
            cv2.waitKey(0)

        return areas


    def dect_gratch_by_lowbound(self, upirtr_tuanhua):

        areas = []

        th = 200

        gray = cv2.cvtColor(upirtr_tuanhua, cv2.COLOR_BGR2GRAY)
        # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的点
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        print('gratch最亮的点：', maxVal)

        # 0) 直接通过阈值二值化
        if maxVal < th:
            return areas
        _, bin_by_th = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

        if self.debug:
            cv2.imshow('white binary', bin_by_th)

        # 1) 将黑色区域通过二值化全部赋值为0, 前提是有黑色区域，不然上面OTSU阈值二值化就不靠谱了
        # 1) 票号一定会显示黑色
        otsu_thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 1) 扩大一点黑色区域
        binary = cv2.erode(binary, tempkernel, iterations=1)
        # 2) 黑色区域0, 覆盖原图, 剩余的进行OTSU阈值分割
        # 2）但是如果没有异常刮擦区域呢，这个二值化得到的结果则拉跨
        gray = cv2.bitwise_and(gray, binary)
        mean = get_gray_mean(gray, cacl_zero=False)
        print('除去黑色，像素均值为：', mean)

        if mean > 190:
            return []

        _, contours, hierarchy = cv2.findContours(bin_by_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.white_thresh:
                areas.append(area)
                cv2.drawContours(upirtr_tuanhua, [cnt], 0, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        print(areas)
        if self.debug:
            cv2.imshow('white contours', upirtr_tuanhua)
            cv2.waitKey(0)
        return areas


    def dect_wihte_area(self, upir_tuanhua):
        """
        把所有黑色除掉，剩余灰度图，再次最大类间方差二值化THRESH_OTSU
        如果是刮擦，那么白上加白
        :param upir_tuanhua:
        :return:
        """
        gray = cv2.cvtColor(upir_tuanhua, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 去噪点?

        # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的点
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        print('最亮的点：', maxVal)
        # 在图像中绘制结果
        cv2.circle(gray, maxLoc, 5, (0, 0, 0), 2)
        cv2.imshow('maxliang', gray)

        # 1) 将黑色区域通过二值化全部赋值为0, 前提是有黑色区域，不然上面OTSU阈值二值化就不靠谱了
        # 1) 票号一定会显示黑色
        otsu_thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print('otsu_thresh', otsu_thresh)
        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 1) 扩大一点黑色区域
        binary = cv2.erode(binary, tempkernel, iterations=1)

        # 2) 黑色区域0, 覆盖原图, 剩余的进行OTSU阈值分割
        # 2）但是如果没有异常刮擦区域呢，这个二值化得到的结果则拉跨
        gray = cv2.bitwise_and(gray, binary)

        # 3) 最大类间方差确定阈值(不统计0)
        # thre = _var_max_interclass(gray, calc_zero=False)
        thre = otsu_threshold(gray)
        print('gratch gray val > ', thre)
        # show 2 提取的刮擦区域带有噪点 通过膨胀操作
        _, binary = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)
        binary = cv2.dilate(binary, tempkernel, iterations=1)
        binary = cv2.erode(binary, tempkernel, iterations=1)

        # 局部二值化不适合
        # th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # cv2.imshow('adapt', th2)

        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.white_thresh:
                areas.append(area)
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        if self.debug:
            cv2.imshow('gray', gray)
            cv2.imshow('binary', binary)
            cv2.waitKey(0)

        return areas


#  最大类间方差确定阈值 输入灰度图
def _var_max_interclass(img, calc_zero=True):
    h, w = img.shape[:2]
    mean = np.mean(img)

    counts = np.zeros((256,), dtype=int)
    prob = np.zeros((256,), dtype=float)

    blank_pix_cnt = 0
    count_map = Counter(img.ravel())
    for pix_val, count in count_map.items():
        if not calc_zero and pix_val == 0:
            blank_pix_cnt += 1
            continue
        counts[pix_val] = count
    prob = counts / (h * w -blank_pix_cnt)
    print(counts[-25:])

    ks = [i for i in range(1, 255)]
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

        if p1 == 0 or p2 == 0:
            continue

        var = (mean*p1 - mean1)**2 / (p1*p2)

        if var > maxvar:
            maxvar = var
            threshold = k

    return threshold


def otsu_threshold(im):
    height, width = im.shape
    pixel_counts = np.zeros(256)
    for x in range(width):
        for y in range(height):
            pixel = im[y][x]
            pixel_counts[pixel] = pixel_counts[pixel] + 1
    # 得到图片的以0-255索引的像素值个数列表
    pixel_counts[0] = 0  # =============
    s_max = (0, -10)
    for threshold in range(256):
        # 遍历所有阈值，根据公式挑选出最好的
        # 更新
        w_0 = sum(pixel_counts[:threshold])  # 得到阈值以下像素个数
        w_1 = sum(pixel_counts[threshold:])  # 得到阈值以上像素个数

        # 得到阈值下所有像素的平均灰度
        u_0 = sum([i * pixel_counts[i] for i in range(0, threshold)]) / w_0 if w_0 > 0 else 0

        # 得到阈值上所有像素的平均灰度
        u_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # 总平均灰度
        u = w_0 * u_0 + w_1 * u_1

        # 类间方差
        g = w_0 * (u_0 - u) * (u_0 - u) + w_1 * (u_1 - u) * (u_1 - u)

        # 类间方差等价公式
        # g = w_0 * w_1 * (u_0 * u_1) * (u_0 * u_1)

        # 取最大的
        if g > s_max[1]:
            s_max = (threshold, g)
    return s_max[0]


def get_gray_mean(gray_img, cacl_zero):
    height, width = gray_img.shape

    pix_sum = 0
    pix_cnt = 0

    for x in range(width):
        for y in range(height):
            pixel = gray_img[y][x]

            if pixel == 0 and not cacl_zero:
                continue
            pix_sum += pixel
            pix_cnt += 1

    return pix_sum //pix_cnt



if __name__ == '__main__':
    bloom_path = r'../hanghui/img_tuanhua/tuanhua_uv_4.jpg'  # 这个最好了
    # upir_tuanhua = r'../hanghui/img_tuanhua/test_ir.jpg'
    # upir_tuanhua = r'../hanghui/img_tuanhua/test_ir_1.jpg'

    upir_tuanhua = r'../hanghui/img_tuanhua/test_ir_bai_4.jpg'  # 可以
    upir_tuanhua = r'../hanghui/img_tuanhua/test_ir_bai_2.jpg'

    # upir_tuanhua = r'gray_white.jpg'


    img = cv2.imread(upir_tuanhua)
    tm = TuanhuaMeasure(bloom_path)
    tm.debug = True

    # blank_list = tm.dect_blank_areas(img)
    # print(blank_list)
    # m.dect_wihte_area(img)

    tm.dect_gratch_by_lowbound(img)
