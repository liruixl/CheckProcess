
import cv2
import numpy as np
from collections import Counter
from tools.imutils import cv_imread, separate_color

"""
团花大小约:(369*145)
以右上角为参考：团花位置距离右侧25，距离上10
以此可大致确定团花位置
可作为常量使用
"""
TH_W = 374
TH_H = 145
TH_XOFF_RIGHT = 25
TH_YOFF_TOP = 10


def paste_to_canvas(dst_h, dst_w, y_off, x_off, img):
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)  # 扩大20像素
    h, w = img.shape
    dst[y_off:y_off+h, x_off:x_off+w] = img
    return dst

class TmpRegionInfo:
    def __init__(self, xmin, ymin, xmax, ymax, oriimg):
        """
        :param xmin: 记录在自身在原图的坐标区域
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
    def __init__(self, upg_img, upir_img, upirtr_img, upuv_img, bbox=None):
        """

        :param upuv_img: 紫外反射图像, 检测黑色涂改墨团
        :param upirtr_img: 红外透视图像, 检测刮擦
        :param upg_img: 可见光图像, 得到文字掩膜
        :param bbox: 团花坐标，若None，使用先验坐标
        """
        check_h, check_w, check_c = upuv_img.shape

        xmin = check_w - TH_W - TH_XOFF_RIGHT
        ymin = TH_YOFF_TOP
        xmax = xmin + TH_W
        ymax = ymin + TH_H

        if bbox != None:
            assert len(bbox) == 4
            xmin,ymin,xmax,ymax = bbox
            assert check_w >= xmax > xmin >= 0 and check_h > ymax > ymin >= 0

        self.upg_box = upg_img[ymin:ymax, xmin:xmax]
        self.upir_box = upir_img[ymin:ymax, xmin:xmax]
        self.upirtr_box = upirtr_img[ymin:ymax, xmin:xmax]
        self.upuv_box = upuv_img[ymin:ymax, xmin:xmax]

        self.text_mask, _ = separate_color(self.upg_box, color="black")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.text_mask = cv2.dilate(self.text_mask, kernel, iterations=1)


class TuanhuaMeasure:
    """
    团花大小约:(369*145)
    以右上角为参考：团花位置距离右侧25，距离上10
    以此可大致确定团花位置
    """

    def __init__(self, check_dir):
        self.debug = False

        self.th_w = 374
        self.th_h = 145
        self.off_x_by_right = 25
        self.off_y_by_top = 10

        TEMP_DIR = check_dir

        upuv = TEMP_DIR + '/' + 'Upuv.bmp'
        upg = TEMP_DIR + '/' + 'UpG.bmp'
        upir = TEMP_DIR + '/' + 'Upir.bmp'
        upirtr = TEMP_DIR + '/' + 'Upirtr.bmp'

        upuv_img = cv_imread(upuv)
        upg_img = cv_imread(upg)
        upir_img = cv_imread(upir)
        upirtr_img = cv_imread(upirtr)

        self.th_template = TuanHuaROI(upg_img,upir_img, upirtr_img, upuv_img)

        self.im = self.th_template.upuv_box.copy()

        # template region  团花左侧区域 中心74
        self.tr_1 = TmpRegionInfo(5, 58 - 20, 30 + 30, 90 + 20, self.im)

        # 团花右侧区域
        # self.tr_1 = TmpRegionInfo(325, 74-50, 325+50, 74+50, self.im)

        # 团花上部区域 y[6, 30]超过三十可能带有文字区域了 中心x187
        # self.tr_1 = TmpRegionInfo(187 - 50, 6, 187 + 50, 30, self.im)

        cv2.imshow('tmp_1', self.tr_1.tmp_img)


        # 1)颜色提取
        self.hsv_mask, self.hsv = separate_color(self.im, color='tuanhua_green')
        # 2)二值化
        # _, self.hsv_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        cv2.imshow('tuanhua hsv', self.hsv_mask)
        cv2.waitKey(0)


        # hsv -> gray -> denoise -> bin，转化为灰度图，然后降噪，在转变为二值图去做最后的匹配
        # 这一步去除一些极小的噪点
        hsv_gray = cv2.cvtColor(self.hsv, cv2.COLOR_BGR2GRAY)
        hsv_gray_denoise = cv2.fastNlMeansDenoising(hsv_gray, h=5.0)
        th, hsv_gray = cv2.threshold(hsv_gray_denoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 固定阈值效果不好
        # _, hsv_gray = cv2.threshold(hsv_gray_denoise, 150, 255, cv2.THRESH_BINARY)
        # 局部二值化 效果不好
        # hsv_gray = cv2.adaptiveThreshold(hsv_gray_denoise, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, C= 0)

        print("去噪后OSTU二值化的阈值：", th)



        self.hsv_mask = hsv_gray.copy()

        cv2.imshow('denoise hsvmask', hsv_gray)
        cv2.waitKey(0)

        gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
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

        # assert count == 1, 'Non-standard group flower image template'

        # 以右上角为原点，左x，下y，这个区域是文字区域
        self.ignore_x_range = (110, 310)
        self.ignore_y_range = (80, 160)

        # 以团花为参考系，文字区域和收款行区域
        # self.num_box = [90, 70, 290, 120]  # xmin 90 100
        self.num_box = [80, 70, 290, 120]
        self.recvbank_box = (0, 90)  # (xmin, ymin) ymin 90 100

        self.blank_thresh = 1
        self.white_thresh = 3

    def measure(self, th_dect:TuanHuaROI, bnd_box):
        """

        :param dect_img: dectected bloom box img
        :param upirtr_img_path: upir check img
        :param bnd_box: [xmin, ymin, xmax, ymax]
        :return: confidence
        """

        xmin, ymin, xmax, ymax = bnd_box

        upg_box = th_dect.upg_box
        upuv_box = th_dect.upuv_box
        upir_box = th_dect.upir_box
        upirtr_box = th_dect.upirtr_box

        upg_black_mask, upg_black = separate_color(upg_box, color='black')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        upg_black_mask = cv2.dilate(upg_black_mask, kernel, iterations=1)



        # 模板匹配左半部分
        result = cv2.matchTemplate(upuv_box, self.tr_1.tmp_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        t_left = max_loc  # (x,y) 访问 result[y][x]
        det_xmin, det_ymin = t_left
        print('模板在检测图上的位置：', t_left)

        # 验证模板匹配准确性 验证通过ok
        # cv2.rectangle(upuv_box, (det_xmin, det_ymin), (det_xmin + 25, det_ymin + 32), (0,0,0))
        # cv2.imshow('pipei', upuv_box)
        # cv2.waitKey(0)

        H, W = self.im.shape[:2]
        canvas = np.zeros((H + 20, W + 20), dtype=np.uint8)  # 扩大20像素
        canvas[10: H+10, 10:W+10] = self.hsv_mask
        cv2.imshow('canvas', canvas)

        # 根据upuv_box V通道的平均值来判断，hsv阈值
        # def v_mean(bgrimg):
        #     hsvimg =

        det_hsv_mask, det_hsv = separate_color(upuv_box, color='tuanhua_green')

        tuanhua_hsv = cv2.cvtColor(upuv_box, cv2.COLOR_BGR2HSV)
        th_v = tuanhua_hsv[:, :, 2]
        V_mean = np.mean(th_v)

        if V_mean > 125:
            print('团花亮度: 太亮')
            det_hsv_mask, det_hsv = separate_color(upuv_box, color='tuanhua_green_v90')

        cv2.imshow('dect tuanhua box', upuv_box)
        cv2.imshow('detect tuanhua mask', det_hsv_mask)

        # 对检测图像也进行少许变造
        hsv_gray = cv2.cvtColor(det_hsv, cv2.COLOR_BGR2GRAY)
        hsv_gray_denoise = cv2.fastNlMeansDenoising(hsv_gray, h=10.0)
        _, hsv_gray = cv2.threshold(hsv_gray_denoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        det_hsv_mask = hsv_gray.copy()
        cv2.imshow('detect tuanhua mask 2', det_hsv_mask)
        cv2.waitKey(0)

        det_H, det_W = upuv_box.shape[:2]
        det_canvas = np.zeros((H + 20, W + 20), dtype=np.uint8)

        dx = det_xmin - self.tr_1.xmin
        dy = det_ymin - self.tr_1.ymin
        print('图像与模板偏移量:', dx, dy)

        det_canvas[10-dy:10-dy+det_H, 10-dx:10-dx+det_W] = det_hsv_mask
        cv2.imshow('canvas det', det_canvas)
        cv2.waitKey(0)


        # XOR结果
        xor_mask = cv2.bitwise_xor(canvas, det_canvas)

        A = np.zeros((H + 20, W + 20), dtype=np.uint8)  # 扩大20像素
        A[10-dy:10-dy+det_H, 10-dx:10-dx+det_W] = upg_black_mask
        A = cv2.bitwise_not(A)
        xor_mask = cv2.bitwise_and(xor_mask, A)


        """
        xor_mask 图像上做检测
        """

        text_mask_1 = cv2.bitwise_not(self.th_template.text_mask)
        text_mask_1 = paste_to_canvas(H+20, W+20, 10, 10, text_mask_1)

        text_mask_2 = cv2.bitwise_not(th_dect.text_mask)
        text_mask_2 = paste_to_canvas(H+20, W+20, 10-dy, 10-dx, text_mask_2)

        text_mask_all = cv2.bitwise_and(text_mask_1, text_mask_2)
        xor_mask_2 = cv2.bitwise_and(xor_mask, text_mask_all)

        # 去噪
        # xor_mask_2 = cv2.fastNlMeansDenoising(xor_mask_2, h = 40.0)

        cv2.imshow('det xor template', np.vstack([xor_mask, xor_mask_2]))
        cv2.waitKey(0)

        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ero = cv2.erode(xor_mask_2, tempkernel, iterations=2)

        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ero = cv2.dilate(ero, tempkernel, iterations=2)

        ero_show = ero.copy()
        ero_show = cv2.cvtColor(ero_show, cv2.COLOR_GRAY2BGR)

        _, contours, hierarchy = cv2.findContours(ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if area > 10:
                # cv2.drawContours(ero_show, [cnt], 0, (0,0,244))
                cv2.rectangle(ero_show, (x,y), (x+w, y+h), (0,0,200))


        cv2.imshow('dila', ero_show)
        cv2.waitKey(0)



        if self.debug:
            print('detected tuanhua:')
            cv2.imshow('tuanhua uv', upuv_box)
            cv2.waitKey(0)


        # 利用红外反射图像上检测是否有涂改（黑），刮擦（白）
        blank_list = self.dect_blank_areas(upir_box)
        blank_list.sort(reverse=True)

        temp = sum(blank_list)
        
        if temp > 150:
            return 0.11
        if temp > 100:
            return 0.21
        if temp > 50:
            return 0.31

        white_list = self.dect_gratch_by_lowbound(upirtr_box)
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

        # 为了更好的检测，膨胀一下
        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.dilate(binary, tempkernel, iterations=1)

        # cv2.imshow('text mask', np.vstack([mask, binary]))

        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        img = upir_tuanhua.copy()
        areas = []
        print('detect blank areas:', len(contours), '个')
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.blank_thresh:
                areas.append(area)
                cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2, lineType=cv2.LINE_AA)


        if self.debug:
            print('Blank areas:', areas)

        if self.debug:
            # cv2.imshow('mask', np.hstack((num_mask, bankname_mask)))
            cv2.imshow('B binary', np.vstack((gray, binary)))  # tuanhua gray and blank area
            cv2.imshow('B detect', img)
            cv2.waitKey(0)

        return areas


    def dect_gratch_by_lowbound(self, upirtr_tuanhua):

        areas = []

        th = 240

        gray = cv2.cvtColor(upirtr_tuanhua, cv2.COLOR_BGR2GRAY)
        # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的点
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

        # gray = cv2.equalizeHist(gray)
        # cv2.imshow('[Hist duibi]', gray)
        # cv2.waitKey(0)

        # img_norm = cv2.normalize(gray, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
        # cv2.imshow("[Norm]", img_norm)
        # cv2.waitKey(0)

        if self.debug:
            print('gratch最亮的点：', maxVal)

        # 0) 直接通过阈值二值化
        if maxVal < th:
            print('maxval < th, not have gratch')
            return areas
        _, bin_by_th = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

        if self.debug:
            cv2.imshow('W binary', bin_by_th)

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

        if self.debug:
            print('除去黑色，像素均值为：', mean)

        if mean > 200:
            print('too bright to judege...')
            return []

        _, contours, hierarchy = cv2.findContours(bin_by_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        show_img = upirtr_tuanhua.copy()
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.white_thresh:
                areas.append(area)
                cv2.drawContours(show_img, [cnt], 0, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        print(areas)
        if self.debug:
            cv2.imshow('W contours', show_img)
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
                # cv2.drawContours(img, [cnt], 0, (0, 0, 255), 2, lineType=cv2.LINE_AA)

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

def exp():
    bloom_path = r'../hanghui/img_tuanhua/tuanhua_uv_4.jpg'  # 这个最好了
    # upir_tuanhua = r'../hanghui/img_tuanhua/test_ir.jpg'
    # upir_tuanhua = r'../hanghui/img_tuanhua/test_ir_1.jpg'

    upir_tuanhua = r'../hanghui/img_tuanhua/test_ir_bai_4.jpg'  # 可以
    upir_tuanhua = r'../hanghui/img_tuanhua/test_ir_bai_2.jpg'

    # upir_tuanhua = r'gray_white.jpg'

    img = cv2.imread(upir_tuanhua)
    tm = TuanhuaMeasure(bloom_path)
    tm.debug = True

    blank_list = tm.dect_blank_areas(img)
    print(blank_list)

    tm.dect_gratch_by_lowbound(img)
    pass

if __name__ == '__main__':


    import os

    # bloom_path = r'../hanghui/img_tuanhua/tuanhua_uv_4.jpg'  # 这个最好了
    bloom_path = r'../tuanhua/tuanhua_template_img/normal_tuanhua_01.jpg'


    tm_yellow = TuanhuaMeasure(r'../tuanhua/20200828_15-39-29')

    tm_v126 = TuanhuaMeasure(r'../tuanhua/20200828_14-45-08_v128')

    tm_yellow.debug = False
    tm_v126.debug = False

    tuanhuayc = r'E:\异常图0927\郭南异常细分\团花异常'
    tuanhuayc = r'E:\异常图0927\敦南\敦南异常'


    # tuanhuayc = r'E:\异常图0927\敦南\敦南正常'
    #
    # tuanhuayc = r'../tuanhua'


    names = os.listdir(tuanhuayc)
    names = [n for n in names if '团花' in n]  # 116对不齐

    # tuanhuayc = r'E:\异常图0927\敦南\敦南正常'
    # names = ['20200828_15-06-02 太淡']

    # print(names)

    for na in names:
        print('processing ', na)
        imgdir = os.path.join(tuanhuayc, na)
        upuv = imgdir + '/' + 'Upuv.bmp'

        upuv_img = cv_imread(upuv)
        h, w, c = upuv_img.shape

        xmin = w - TH_W - TH_XOFF_RIGHT
        ymin = TH_YOFF_TOP
        xmax = xmin + TH_W
        ymax = ymin + TH_H

        upg = imgdir + '/' + 'UpG.bmp'
        upir = imgdir +'/' + 'Upir.bmp'
        upirtr = imgdir + '/' + 'Upirtr.bmp'



        upg_img = cv_imread(upg)
        upir_img = cv_imread(upir)
        upirtr_img = cv_imread(upirtr)

        th_dect = TuanHuaROI(upg_img, upir_img, upirtr_img, upuv_img, [xmin, ymin, xmax, ymax])

        tuanhua_hsv = cv2.cvtColor(th_dect.upuv_box, cv2.COLOR_BGR2HSV)
        th_h = tuanhua_hsv[:,:,0]
        th_s = tuanhua_hsv[:,:,1]
        th_v = tuanhua_hsv[:,:,2]


        v_mean = np.mean(th_v)

        print("tuanhua box hsv mean:", np.mean(th_h), np.mean(th_s), v_mean)



        score_1 = tm_yellow.measure(th_dect, [xmin, ymin, xmax, ymax])
        score_2 = tm_v126.measure(th_dect, [xmin, ymin, xmax, ymax])


        print('得分：',[score_1, score_2])


