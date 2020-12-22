import sys

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tools.imutils import cv_imread, separate_color

from red_waterline.union_find import UnionFind
from red_waterline.cu_line import detect_darkline




def bloom2_debug():
    # 变造红水线数据
    # check_dir = r'E:\DataSet\redline\UpG_redline'
    # debug_dir = r'E:\DataSet\redline\debug_dark_line_midu'

    # 正常红水线数据
    check_dir = r'E:\DataSet\redline_ok\redline_normal'
    debug_dir = r'E:\DataSet\redline_ok\debug_dark_line_midu_2'

    filename_list = os.listdir(check_dir)
    for imgname in filename_list:

        print('process:', imgname)

        imagePath = os.path.join(check_dir, imgname)
        img = cv_imread(imagePath)

        iimg = img.copy()

        res_tuan = detect_bloom2(iimg)
        res_darkline = detect_darkline(iimg)

        # res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(debug_dir, imgname), np.vstack((res_tuan,res_darkline)))


# 注意返回值，实验时就返回结果图像了
def detect_bloom2(img, debug=False):


    img_bgr = img.copy()  # 用于绘制

    red_much = False

    mask_red, redhsv = separate_color(img_bgr, color='red')
    mask_yz, yingzhang = separate_color(img_bgr, color='yingzhang')
    mask_yz = cv2.bitwise_not(mask_yz)

    redline = cv2.bitwise_and(redhsv, redhsv, mask=mask_yz)
    # redline = redhsv
    # ==================================================
    gray = cv2.cvtColor(redline, cv2.COLOR_BGR2GRAY)  # 打算用灰度图去判断深色的晕染线。。
    # ==================================================

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # 判断红水线像素数，根据数量大致判断整体的颜色程度

    cnt = 0
    h, w = binary.shape
    for i in range(h):
        for j in range(w):
            if binary[i][j] > 10:
                cnt += 1

    print('range should be in ', w * 10, 2 * w * 10)
    print('红水线比例：', cnt, '/', h * w)

    tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    ero1 = cv2.erode(binary, tempkernel, iterations=1)

    tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    ero2 = cv2.erode(ero1, tempkernel, iterations=1)

    if cnt < 2*w*10:
        ero2 = ero1
        red_much = False
    else:
        red_much = True

    # ero2 图可以作为初步判断依据，晕染外的区域基本只剩噪点了
    # 但是对于正常红水线来说，颜色较深的也可能被误检，mmp
    # ====================================11111=====================================
    blur_avg = cv2.blur(ero2, (11, 11))
    thre = (4 * 4 * 255) // (11 * 11)
    _, blur_bin = cv2.threshold(blur_avg, thre-10, 255, cv2.THRESH_BINARY)
    # _, blur_bin = cv2.threshold(blur_avg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 只寻找最外轮廓
    _, contours, hierarchy = cv2.findContours(blur_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    CONTOURS_1 = contours  # 保留一下，以防用到
    RECTS_1 = []
    AREA_1 = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        RECTS_1.append([x,y,x+w,y+h])
        AREA_1.append(area)

    if debug:
        print('find contours from avg blur:', len(RECTS_1))

    # N1个检测框
    N1 = len(RECTS_1)
    uf = UnionFind(N1)
    for i in range(0, N1-1):
        for j in range(i+1, N1):
            xmin1, ymin1, xmax1, ymax1 = RECTS_1[i]
            xmin2, ymin2, xmax2, ymax2 = RECTS_1[j]

            l_x = max(xmin1, xmin2)
            r_x = min(xmax1, xmax2)
            t_y = max(ymin1, ymin2)
            b_y = min(ymax1, ymax2)

            # 阈值, 作为合并的依据。
            if (r_x - l_x >= 0) and (b_y - t_y >=0):
                uf.union(i,j)
            elif b_y - t_y > 0 and abs(r_x - l_x) <= 15:
                uf.union(i,j)
            elif r_x - l_x > 0 and abs(b_y - t_y) <= 10:
                uf.union(i,j)
            elif (r_x - l_x < 0) and (b_y - t_y < 0):
                if abs(r_x - l_x) <= 15 and abs(b_y - t_y) <= 10:
                    uf.union(i,j)

    # 合并idx结束
    # 处理合并的idx
    rect_merge = []
    area_merge = []
    D= {}
    for idx in range(0, N1):
        p = uf.find(idx)
        if p in D.keys():
            D[p].append(idx)
        else:
            D[p] = []
            D[p].append(p)

    if debug:
        print('union find:',D)
    for k, rects in D.items():
        if(len(rects) == 1):
            rect_merge.append(RECTS_1[rects[0]])
            xmin, ymin, xmax, ymax = RECTS_1[rects[0]]
            area_merge.append((xmax-xmin)*(ymax-ymin))
        else:
            xmin = min([RECTS_1[r][0] for r in rects])
            ymin = min([RECTS_1[r][1] for r in rects])
            xmax = max([RECTS_1[r][2] for r in rects])
            ymax = max([RECTS_1[r][3] for r in rects])

            rect_merge.append([xmin,ymin,xmax,ymax])
            area_merge.append((xmax-xmin)*(ymax-ymin))
    if debug:
        print('avgblur合并后的框:',rect_merge)
        print('avgblue合并后面积:',area_merge)




    # 得到初步结果rect_merge，但是会误检正常的粗的红水线,而且很多，nmsl,mmp
    # 所以这一步得到的只能是候选结果
    # 根据面积筛选出可能为红水线的候选框:RES_1
    # ================================================================
    RES_1 = []  # 可能为红水线的候选框，正常的图像上误检较多
    # ================================================================

    for r in rect_merge:
        xmin, ymin, xmax, ymax = r
        if xmax - xmin < 15 or ymax - ymin < 15:
            # 面积小的绘制为蓝色
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (255, 200, 0), 1)
        else:
            # 大的绘制为红色 并加入到maybe候选框
            cv2.rectangle(img_bgr, (xmin,ymin),(xmax,ymax),(0,0,255),1)
            RES_1.append(r)

    if debug:
        print('根据面积(15,15)过滤后的候选框RES_1:', RES_1)
        print('========================================')



    if red_much is False:
        if debug:
            cv2.imshow('red little', img_bgr)
            cv2.waitKey(0)

        return rect_merge


    """
    如果红水线颜色深，则继续判断
    否则直接返回第一步的结果rect_merge
    """
    # 再次进行腐蚀（竖着的），膨胀运算，留下团状晕染的区域
    # 如果有轮廓，那么必定是晕染

    #  2. 再次去做腐蚀ero2的操作，这次得到的大轮廓一定是晕染
    tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    ero3 = cv2.erode(ero2, tempkernel, iterations=1)

    tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    res_2 = cv2.dilate(ero3, tempkernel, iterations=1)

    # res 图求轮廓，可以求得一般的晕染（就一团红色，但对于色深线的还是不行ero2应该行）
    # 但是对于正常红水线来说，颜色较深的也可能被误检
    # ======================================22222===================================

    _, contours, hierarchy = cv2.findContours(res_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    RECTS_2 = []
    RECTS_2_TINY = []  # 过滤掉的矩形
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w < 15 or h < 15 or area < 200:  # 这里又是阈值
            if w < 3 or h < 3 or area < 9:  # 太小的忽略
                pass
            else:
                RECTS_2_TINY.append([x,y, x+w, y+h])
            continue
        RECTS_2.append([x,y, x+w, y+h])
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 1)

    if debug:
        print('再次腐蚀后，一定是晕染的区域RECTS_2:', RECTS_2)
        print('太小过滤掉的框RECTS_2_TINY::', len(RECTS_2_TINY), RECTS_2_TINY)

    def in_rect(cx,cy, rect):
        assert len(rect) == 4
        xmin,ymin,xmax,ymax = rect
        return xmin<=cx<=xmax and ymin<= cy <= ymax


    # 一般来说，RECT_2是 RES_1的子集
    # 那么RECT_2 由子集中的某些构成，如果子集组成的面积与候选RECT_2相当，认为是晕染
    RES_1_FLAG = [0 for _ in range(0, len(RES_1))]
    RES_1_PART = [[] for _ in range(0, len(RES_1))]
    RES_1_PART_AREA = [0 for _ in range(0, len(RES_1))]

    for r2 in RECTS_2:
        x1,y1,x2,y2 = r2
        rect2_area = (x2 - x1)*(y2 - y1)
        for i in range(0, len(RES_1)):
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            if in_rect(cx,cy,RES_1[i]):
                RES_1_FLAG[i] += 1
                RES_1_PART[i].append(r2)  # 由于子集关系，可以认为由r2组成
                RES_1_PART_AREA[i] += rect2_area  # 子集总面积
    # ==============================================================
    RES_2 = []  # 根据规则保留RES_1中的结果，有的可能框大了，由于合并操作
    # ==============================================================
    for i in range(0, len(RES_1)):
        x1, y1, x2, y2 = RES_1[i]
        r1_area = (x2 - x1) * (y2 - y1)
        # 这又是个阈值啊啊啊啊
        if RES_1_FLAG[i] != 0 and (RES_1_PART_AREA[i] / r1_area > 0.15):
            RES_2.append(RES_1[i])
            RES_1_FLAG[i] = -1  # 已经确认保留的，后续无需再验证, 置为-1

            x1,y1,x2,y2 = RES_1[i]
            # cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if debug:
        print('根据腐蚀图依然有大轮廓，RES_1中可以认为是晕染的区域RES_2:',RES_2)

    for i in range(0, len(RES_1)):
        if RES_1_FLAG[i] == -1:
            continue
        for r2_tiny in RECTS_2_TINY:
            x1, y1, x2, y2 = r2_tiny
            r2_tiny_area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if in_rect(cx,cy,RES_1[i]):
                RES_1_FLAG[i] += 1  # 个数 += 1
                RES_1_PART_AREA[i] += r2_tiny_area  # 子集总面积 +=

    # 可以利用密度， 个数/候选框总面积 或者   子集总面积/总面积

    RES_3 = []

    for i in range(0, len(RES_1)):
        if RES_1_FLAG == -1:
            continue
        x1, y1, x2, y2 = RES_1[i]
        r1_area = (x2 - x1) * (y2 - y1)
        if debug:
            print('长宽比:', (x2 - x1) / (y2-y1))
        if (x2 - x1) / (y2-y1) > 3.0:
            continue
        if debug:
            print('第', i, '个, 面积所占比例:', RES_1_PART_AREA[i] / r1_area)
        # print('第', i, '个, 个数所占比例:', RES_1_FLAG[i] / (x2-x1))

        if RES_1_PART_AREA[i]/r1_area > 0.05 and RES_1_FLAG[i] > 3:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), 2)
            RES_3.append(RES_1[i])

    if debug:
        print('根据密度可以认为是晕染的区域RES_3:', RES_3)


    if debug:
        cv2.imshow('redline', redline)
        cv2.imshow('separate', np.vstack((gray, binary)))
        cv2.imshow('process', np.vstack((ero1, ero2, blur_avg,blur_bin, ero3, res_2)))
        cv2.imshow('res', img_bgr)
        # plt.hist(gray.ravel(), 255)
        # plt.show()
        cv2.waitKey(0)

    # return RES_2
    # return np.vstack((cv2.cvtColor(blur_bin,cv2.COLOR_GRAY2BGR),
    #                   cv2.cvtColor(res_2, cv2.COLOR_GRAY2BGR),
    #                   img_bgr))  # ero3

    print("返回晕染区域结果：", RECTS_2 + RES_3)
    return RECTS_2 + RES_3   # (xmin, ymin, xmax, ymax)





def bloom_test():
    """
    十月份测试集郭南票中的红水线测试。
    :return:
    """
    red_dir = r'E:\异常图0927\红水线晕染'
    names = os.listdir(red_dir)
    imgpaths = [os.path.join(red_dir, na) for na in names]

    for imgpath in imgpaths:
        img = cv_imread(imgpath)
        detect_bloom2(img, debug=True)


if __name__ == '__main__':

    imagePath = r'E:\异常图0927\红水线晕染\88.jpg'
    img = cv_imread(imagePath)

    # 1.检测晕染，以及刮擦产生的晕染,ok的
    pairs = detect_bloom2(img, debug=True)

    # bloom_test()

    cv2.imshow('imread', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

