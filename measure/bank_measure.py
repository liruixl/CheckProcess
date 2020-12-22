import numpy as np
import cv2
import os
from tools.imutils import cv_imread
from measure.stitcher import Stitcher


"""
行徽坐标信息： 
中心点(1140, 78) (1137, 74)
小农行行徽大小：88*88
大农行行徽大小：96*96
"""
BANK_W = 106
BANK_H = 106
BANK_XOFF_RIGHT = 1140 - BANK_W//2
BANK_YOFF_TOP = 70 - BANK_H//2


def paste_to_canvas(dst_h, dst_w, y_off, x_off, img):
    dst = np.zeros((dst_h, dst_w), dtype=np.uint8)  # 扩大20像素
    h, w = img.shape
    dst[y_off:y_off+h, x_off:x_off+w] = img
    return dst


def var_bank(bank_img):
    '''
    :param bank_img: bgr
    :return:
    '''
    bank_gray = cv2.cvtColor(bank_img, cv2.COLOR_BGR2GRAY)
    _, bank_bin = cv2.threshold(bank_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)





class BankMeasure:
    def __init__(self, icon_path):

        self.debug = False
        self.icon_temp = cv2.imread(icon_path)

        # _, blur_bin = cv2.threshold(blur_avg, thre, 255, cv2.THRESH_BINARY)
        # _, blur_bin = cv2.threshold(blur_avg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        icon_gray = cv2.cvtColor(self.icon_temp, cv2.COLOR_BGR2GRAY)
        _, self.icon_bin = cv2.threshold(icon_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.stitcher = Stitcher(debug=self.debug)

    def measure(self, img):

        off_or_none = self.stitcher.stitch(img, self.icon_temp, showMathes=self.debug)

        if off_or_none is None:
            return 0.0

        off_x, off_y = off_or_none
        # assert off_x > 0 and off_y > 0  # debug

        ori_x = 0
        ori_y = 0

        if off_x < 0:
            ori_x = -off_x
            off_x = max(0, off_x)
        if off_y < 0:
            ori_y = -off_y
            off_y = max(0, off_x)

        H, W = self.icon_temp.shape[:2]
        if ori_x > W or ori_y > H:
            return 0.0

        icon_move = self.icon_bin[ori_x:, ori_y:]
        H, W = icon_move.shape

        aligned = img[off_y : off_y + H, off_x : off_x + W]
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ah, aw = binary.shape
        aligned_duiqi = np.zeros((H, W), dtype=np.uint8)
        aligned_duiqi[0: ah, 0: aw] = binary
        binary = aligned_duiqi

        # 模板icon_bin与aligned对齐，且大小一致了
        # 应该全为0，黑色就对了
        # 但实际上会有噪点 再做一次腐蚀膨胀
        # mask = cv2.bitwise_xor(self.icon_bin, binary)

        move9s = []
        ioff = [-1, 0, 1]
        joff = [-1, 0, 1]

        for i in ioff:
            for j in joff:
                M = np.float32([[1,0,i], [0,1,j]])
                dst = cv2.warpAffine(icon_move, M, (W, H))
                move9s.append(dst)

        def get255cnt(img):
            cnt = 0
            h,w = img.shape
            for i in range(h):
                for j in range(w):
                    if img[i][j] > 2:
                        cnt+=1
            return cnt

        masks = [cv2.bitwise_xor(move_icon, binary) for move_icon in move9s]
        masks_255_cnt = [get255cnt(m) for m in masks]
        minidx = masks_255_cnt.index(min(masks_255_cnt))
        icon_move = move9s[minidx]
        # print('各种偏移白色像素数', masks_255_cnt)


        if self.debug:
            cv2.imshow('move9', np.hstack(masks))
            cv2.imshow('xor', np.vstack((icon_move, binary)))
            cv2.imshow('dst', dst)
            cv2.waitKey(0)

        mask = cv2.bitwise_xor(icon_move, binary)
        conf = self._convert_mask2conf(mask)

        return conf

    def _convert_mask2conf(self, mask):
        """
        :param mask: binary img
        :return: conf
        """
        tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ero = cv2.erode(mask, tempkernel, iterations=1)
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

        areas.sort(reverse=True)

        if self.debug:
            print('bank bad area:', areas)

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

    def measure_with_loc(self, upuv_path, bnd_box):
        img = cv2.imread(upuv_path)
        xmin, ymin, xmax, ymax = bnd_box
        img = img[ymin : ymax, xmin, xmax]
        return self.measure(img)


if __name__ == '__main__':
    bank_temp = r'E:\DataSet\measure_dataset\template\bank_icon\zhongnong_uv.jpg'

    nonghang_1 = r'../bank_icons/01nonghang.jpg'
    nonghang_2 = r'../bank_icons/20nonghang2.jpg'
    nonghang_3 = r'../bank_icons/21nonghang3.jpg'

    bank_1_measure = BankMeasure(nonghang_1)
    bank_2_measure = BankMeasure(nonghang_2)
    bank_3_measure = BankMeasure(nonghang_3)




    test_check_dir = r'E:\异常图0927\敦南\敦南异常'
    # test_check_dir = r'E:\异常图0927\敦南\敦南正常'

    check_ids = os.listdir(test_check_dir)
    check_ids = [id for id in check_ids if '行徽' in id]
    # check_ids = [id for id in check_ids if '15-41-04' in id]


    print('当前文件夹支票数量:', len(check_ids))
    exit(0)
    bank_1_measure.debug = True
    bank_2_measure.debug = True
    bank_3_measure.debug = True


    for na in check_ids:
        imgdir = os.path.join(test_check_dir, na)
        upuv = imgdir + '/' + 'Upuv.bmp'
        upg = imgdir + '/' + 'UpG.bmp'
        upir = imgdir + '/' + 'Upir.bmp'
        upirtr = imgdir + '/' + 'Upirtr.bmp'

        upuv_img = cv_imread(upuv)
        h, w, c = upuv_img.shape

        xmin = w - BANK_W - BANK_XOFF_RIGHT
        ymin = BANK_YOFF_TOP
        xmax = xmin + BANK_W
        ymax = ymin + BANK_H



        bank_img = upuv_img[ymin:ymax, xmin:xmax]

        # cv2.imshow('bank dect', bank_img)
        # cv2.waitKey(0)



        score_1 = bank_1_measure.measure(bank_img)
        cv2.destroyAllWindows()
        score_2 = bank_2_measure.measure(bank_img)
        cv2.destroyAllWindows()
        score_3 = bank_3_measure.measure(bank_img)
        cv2.destroyAllWindows()


        sc = max(score_2, score_1, score_3)
        if sc < 0.5:
            print('processing', na)
            print('   得分：', sc)






