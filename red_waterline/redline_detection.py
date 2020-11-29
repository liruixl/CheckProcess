import sys

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tools.imutils import cv_imread, separate_color

from red_waterline.union_find import UnionFind
from red_waterline.cu_line import detect_darkline



'''
此文件多为实验程序demo
不是最终算法
'''

def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    # 7. 存储中间图片
    # cv2.imwrite("binary.png", binary)
    # cv2.imwrite("dilation.png", dilation)
    # cv2.imwrite("erosion.png", erosion)
    # cv2.imwrite("dilation2.png", dilation2)

    cv2.imshow('preprocess', np.vstack([sobel, dilation2]))

    return dilation2


def findTextRegion(org, img):
    region = []

    # 1. 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    maxContour = 0

    print('find conrours:', len(contours))
    xs, ys, ws, hs = [], [],[],[]
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        x, y, w, h = cv2.boundingRect(cnt)
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

        cv2.rectangle(org, (x, y), (x + w, y + h), (0, 0, 0), 1)

        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxContour = cnt

    cv2.imshow("resize_res", org)
    return xs, ys, ws, hs


def preprocess_redline(gray):

    # 2. 二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数，
    # 红水线核函数的思考,30一般是文字长度，涂改，2*2去除红点点状的
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))

    # 3.腐蚀一次，去除点状红点
    temp = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erosion = cv2.erode(binary, temp, iterations=1)

    blur = cv2.medianBlur(erosion, 5)

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(blur, element2, iterations=1)
    # dilation = binary

    cv2.imshow('preprocess', np.vstack([erosion,blur,dilation]))

    return dilation

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

def constract_brightness(src1, a, g):
    src2 = np.zeros(src1.shape, src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)
    # cv2.namedWindow("constract", cv2.WINDOW_NORMAL)
    # cv2.imshow("constract", dst)
    return dst


# 注意返回值，实验时就返回结果图像了
def detect_bloom2(img, debug=False):

    img_bgr = img.copy()  # 用于绘制
    mask_red, redhsv = separate_color(img_bgr, color='red')
    mask_yz, yingzhang = separate_color(img_bgr, color='yingzhang')
    mask_yz = cv2.bitwise_not(mask_yz)

    redline = cv2.bitwise_and(redhsv, redhsv, mask=mask_yz)
    # redline = redhsv
    # ==================================================
    gray = cv2.cvtColor(redline, cv2.COLOR_BGR2GRAY)  # 打算用灰度图去判断深色的晕染线。。
    # ==================================================

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


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

    # ero2 图可以作为初步判断依据，晕染外的区域基本只剩噪点了
    # 但是对于正常红水线来说，颜色较深的也可能被误检，mmp
    # ====================================11111=====================================
    blur_avg = cv2.blur(ero2, (11, 11))
    thre = (4 * 4 * 255) // (11 * 11)
    _, blur_bin = cv2.threshold(blur_avg, thre, 255, cv2.THRESH_BINARY)
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
        print('合并后的框:',rect_merge)
        print('合并后面积:',area_merge)



    # 得到初步结果rect_merge，但是会误检正常的粗的红水线,而且很多，nmsl,mmp
    # 所以这一步得到的只能是候选结果
    # 根据面积筛选出可能为红水线的候选框:RES_1
    # ================================================================
    RES_1 = []  # 可能为红水线的候选框，正常的图像上误检较多
    # ================================================================

    for r in rect_merge:
        xmin, ymin, xmax, ymax = r
        if xmax - xmin < 15 or ymax - ymin < 15:
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (255, 200, 0), 1)
        else:
            cv2.rectangle(img_bgr,(xmin,ymin),(xmax,ymax),(0,0,255),1)
            RES_1.append(r)

    if debug:
        print('过滤后的候选框;', RES_1)
        print('========================================')

    # 再次进行腐蚀（竖着的），膨胀运算，留下团状晕染的区域
    # 如果有轮廓，那么必定是晕染


    #  2. 再次去做腐蚀操作，这次得到的大轮廓一定是晕染
    tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    ero3 = cv2.erode(ero2, tempkernel, iterations=1)

    # res = cv2.medianBlur(ero2, 5)
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
        print('再次腐蚀是晕染的区域吧:', RECTS_2)
        print('太小过滤掉的框:', len(RECTS_2_TINY), RECTS_2_TINY)

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
        print('根据腐蚀图依然有大轮廓，RES_1中可以认为是晕染的区域:',RES_2)

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


def detect_bloom(img_bgr, debug=False):

    mask, img = separate_color(img_bgr, color='red')

    # 1. 红水线提取部分（背景为全部为黑色0） 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    h, w = img.shape[:2]


    # 2.1 拉伸前做腐蚀操作，将小点腐蚀掉
    tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ero = cv2.erode(gray, tempkernel, iterations=1)
    # ero = cv2.dilate(ero, tempkernel, iterations=1)

    if debug:
        cv2.imshow('gray_pre', np.vstack((gray, ero)))

    # 2.2 膨胀横向补偿每一根红水线。弃用，基本全膨胀完了，由于高度太小的原因
    # temp = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    # gray = cv2.dilate(gray, temp, iterations=1)
    # cv2.imshow('gray_dilate', gray)

    # 3.为了方便后续形态学处理，将高度拉伸
    gray = cv2.resize(ero, (w, h*3))

    # 3. 形态学变换的预处理，突出晕染变造区域
    dilation = preprocess_redline(gray)


    # 4. 红水线，垂直投影统计白色点个数，找异常
    v_projection = np.sum(dilation//255, axis=0)
    if debug:
        plt.plot(v_projection, 'r-')
        plt.show()

    # 5. 阈值
    thed = h*3//3 * 2
    thed = h*3//2  # resize 的阈值,高度的一半
    # thed = h//2 - 10  # 高度的三分之二 还是一半好呢
    print('thred:', thed)

    # 6.1 寻找晕染边界
    col_bounnd_list = []
    # 寻找边界i, 以i附近的均值来看，弃用, 直方图不是连续的
    # 搜索所有大于阈值的idx，搜索边界情况有点多，由于不是单调增或减
    for i, x in enumerate(v_projection[:-10]):
        mean_1 = np.mean(v_projection[i:i+10], dtype=int)
        if mean_1 >= thed:
            col_bounnd_list.append(i + 5)

    # col_bounnd_list.append(w - 1)  # 防止i为奇数，最后也有变造
    # print(col_bounnd_list)

    # 6.2 合并接近的区域
    pairs = []
    idx = 0
    while idx < len(col_bounnd_list):
        left = idx

        while idx+1 < len(col_bounnd_list) and col_bounnd_list[idx+1] - col_bounnd_list[idx] < 5:
            idx += 1
        pairs.append((col_bounnd_list[left], col_bounnd_list[idx]))
        idx += 1

    print(pairs)

    if debug:
        # h, w = img_bgr.shape[:2]
        for x1, x2 in pairs:
            if x2 - x1 > 10:
                cv2.rectangle(img_bgr, (x1, 2), (x2, h - 2), (0, 0, 255), 1)  # red
            else:
                cv2.rectangle(img_bgr, (x1, 2), (x2, h - 2), (255, 0, 0), 1)  # 窄

        cv2.imshow('result', img_bgr)

    return pairs


def detect_whiteink(img, debug=False):
    mask, res = separate_color(img, color='white')

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 目标: 把零散的小白点腐蚀掉, 白墨团区域全搞成白色(非0区域)
    # 问题:白墨团中可能带有汉字, 也可能把墨团腐蚀掉了,区域变小

    # 高斯滤波,会使小白点区域变大
    # mask = cv2.GaussianBlur(mask, (5,5), 1)
    # 中值滤波一次, 带文字的白墨团也会被腐蚀
    # mask = cv2.medianBlur(mask, 5)
    # 均值滤波然后二值化,是否可行,小白点区域的均值肯定小,墨团区域及时有黑色文字部分也肯定大
    # 然后再膨胀一下
    mask = cv2.blur(mask, (7,7))
    thre = (3*3*255) // (7*7)
    _, mask = cv2.threshold(mask, thre, 255, cv2.THRESH_BINARY)

    # 有少量的小白点,腐蚀一次,但是也会把那种白莫团里面有字的腐蚀掉
    # erosion = cv2.erode(mask, element1, iterations=1)

    # 模糊
    # erosion = cv2.GaussianBlur(erosion, (5, 5), 4)

    #  膨胀一下,轮廓明显
    dilation = cv2.dilate(mask, element2, iterations=1)

    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('find white ink conrours:', len(contours))

    xs, ys, ws, hs = [], [], [], []
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积,过滤太小的
        area = cv2.contourArea(cnt)
        if area < 16:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    if debug:
        cv2.imshow('mask_white', np.vstack((mask,  dilation)))  # 白墨团，基本能提取出HSV
        # plt.figure('process')
        # plt.subplot(3,1,1)
        # plt.imshow(mask,cmap ='gray')
        # plt.subplot(3,1,2)
        # plt.imshow(erosion,cmap ='gray')
        # plt.subplot(3,1,3)
        # plt.imshow(dilation,cmap ='gray')
        # plt.show()
        cv2.imshow('white_conrours', img)
    return xs, ys, ws, hs


def detect_scratch(img, debug=False):
    mask, res = separate_color(img, color='gratch')

    # 1.1中值滤波对消除椒盐噪声非常有效，能够克服线性滤波器带来的图像细节模糊等弊端，
    blur = cv2.medianBlur(mask, 3)  # 核尺寸越大,过滤越多的噪点

    # 1.2高斯滤波,对于提取的变造图像,效果没中值滤波好,弃用
    # img = cv2.GaussianBlur(mask,(5,5),2)

    # 2.1 开运算,再次过滤噪点, 可获得刮擦主要区域, 然后可以在去blur连接周围噪点
    rectKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, rectKernel, iterations=1)  # 开运算，腐蚀，膨胀

    # 2.2 先不过滤,直接使用滤波后的图像找轮廓,然后处理联通域
    # opening = blur

    cimg = img  # 用于绘制的图像

    # 3.可以传递 RETR_EXTERNAL 只寻找最外层轮廓, 全部:RETR_TREE
    _, contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(hierarchy.shape)  # (1, count, 4) [Next, Previous, First Child, Parent]
    # print(contours[0].shape)  # (3, 1 , 2) 而不是(3,2), 3是点的个数
    # print(hierarchy)
    print('find gratch conrours:', len(contours))

    # for idx in range(0, len(contours)):
    #     color = (0,255,0)
    #     pts = cv2.approxPolyDP(contours[idx],cv2.arcLength(contours[idx],True)*0.005, True)
    #     cimg = cv2.polylines(cimg,pts,True,color)
    #     # print(pts.shape)  # (n, 1, 2)
    #     cv2.circle(cimg, (pts[0][0][0],pts[0][0][1]), 3, color)
    #     cv2.circle(cimg, (pts[1][0][0],pts[1][0][1]), 2, color)
    #     for i in range(2, len(pts)):
    #         cv2.circle(cimg, (pts[i][0][0], pts[i][0][1]), 1, color)
    #     if idx == 0:
    #         maxrect = cv2.boundingRect(contours[idx])
    #     else:
    #         rect = cv2.boundingRect(contours[idx])
    #         # maxrect = cv2.max(rect, maxrect) # cvMaxRect呢
    #
    # x, y, w, h = maxrect
    # cv2.rectangle(cimg, (x, y), (x + w, y + h), (0, 0, 255), 1)

    rects = []
    havemerge = []
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积,过滤太小的
        area = cv2.contourArea(cnt)
        rects.append(cv2.boundingRect(cnt))

    for i in range(len(rects)):
        if i in havemerge:
            continue


    xs, ys, ws, hs = [], [], [], []
    # 4.合并联通域, 基于两个轮廓的距离, 点到轮廓的距离
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积,过滤太小的
        area = cv2.contourArea(cnt)

        x, y, w, h = cv2.boundingRect(cnt)
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

        cv2.rectangle(cimg, (x, y), (x + w, y + h), (0, 0, 255), 1)
    if debug:
        cv2.imshow('mask', mask)
        cv2.imshow('blur', blur)
        cv2.imshow('opening', opening)
        cv2.imshow('draw_mask', cimg)


def detect_blackink(img,debug = False):
    # (UpG_161314,UpG_161319)

    mask, res = separate_color(img, color='black')
    cimg = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

    element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ele = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 1.第一次去做小腐蚀吗?  目的太粗的字体搞细一点,
    # 产生噪点 和 一些粗体留下来的小团
    temp = cv2.erode(mask, ele, iterations=1)
    cv2.imshow('first ero', temp)

    # 2.去除小的噪点,还剩小团
    blur = cv2.medianBlur(temp, 7)
    # blur = cv2.medianBlur(blur, 5)


    #  3.腐蚀小团,参数选取3*3的, 再中值滤波可去除小团,
    #   如果黑墨团也小的话, 无能为力..那就两次腐蚀?
    erosion = cv2.erode(blur, element1, iterations=1)
    erosion = cv2.medianBlur(erosion, 5) # 防止墨团里面有黑点,再去腐蚀
    # 3.1 两次腐蚀blur操作,,腐蚀没了
    erosion = cv2.erode(erosion, element1, iterations=1)
    erosion = cv2.medianBlur(erosion, 5)

    # 3.1 感觉还可以在这里提出小的团,根据轮廓大小,做一个mask?????

    #  4.再次膨胀,去找轮廓
    dilation = cv2.dilate(erosion, element1, iterations=2)

    cv2.imshow('mask', mask)

    cv2.imshow('blur', blur)
    cv2.imshow('fushi', erosion)
    cv2.imshow('pengzhang', dilation)

    # 检测⚪不好使,弃用
    # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 2, 20,
    #                             param1=50,param2=30, minRadius=0, maxRadius=30)
    # print(circles.shape)
    #
    # # circles = np.uint16(circles)
    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow('detected circles', cimg)


def bloom_demo():
    imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161615.jpg'
    img = cv_imread(imagePath)

    pair = detect_bloom(img)

    h, w = img.shape[:2]
    for x1, x2 in pair:
        cv2.rectangle(img, (x1, 2), (x2, h - 2), (0, 0, 255), 1)

    cv2.imshow('result', img)
    cv2.waitKey(0)


def bloom_debug():
    # 变造红水线
    # check_dir = r'E:\DataSet\redline\UpG_redline'
    # debug_dir = r'E:\DataSet\redline\debug_bloom_th5'

    # 正常红水线
    check_dir = r'E:\DataSet\redline_ok\redline_normal'
    debug_dir = r'E:\DataSet\redline_ok\debug_1'

    filename_list = os.listdir(check_dir)
    for imgname in filename_list:

        print('process:', imgname)

        imagePath = os.path.join(check_dir, imgname)
        img = cv_imread(imagePath)

        mask, res = separate_color(img, color='red')
        pair = detect_bloom(res)

        h, w = img.shape[:2]
        for x1, x2 in pair:
            if x2 - x1 > 10:
                cv2.rectangle(img, (x1, 2), (x2, h - 2), (0, 0, 255), 1)  # red
            else:
                cv2.rectangle(img, (x1, 2), (x2, h - 2), (255, 0, 0), 1)  # 窄

        cv2.imwrite(os.path.join(debug_dir, imgname), img)


def gratch_demo():
    imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161328.jpg'
    img = cv_imread(imagePath)

    detect_scratch(img, debug=True)


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
    # 黑墨团  # (UpG_161314,UpG_161319,UpG_161324++, UpG_161333,UpG_161342, UpG_161347<,UpG_161405++,UpG_161436,
    # UpG_161451, UpG_161500,UpG_161505 blue, UpG_161510, UpG_161528, UpG_161533, UpG_161537 font, UpG_161601,
    # UpG_161605--, UpG_161615, UpG_161619++, UpG_161624, UpG_161628, UpG_161642,UpG_161652,UpG_161701,UpG_161711,
    # UpG_161729, UpG_161739, UpG_161753, UpG_161831,UpG_161840,UpG_161845,UpG_161849,UpG_161854,UpG_161858,
    # UpG_161903)

    # UpG_161657 font整, UpG_161706 font整, UpG_161743 font粗,UpG_161748 font壹, UpG_161757 font,UpG_161812 font,

    # resize垂直投影，效果不太好
    # 晕染 #(UpG_161314 >,UpG_161324 >, ,UpG_161701,UpG_161351>, UpG_161510>>,)
    # error (UpG_161337 没误检了, UpG_161414 加入一次膨胀,UpG_161505蓝色, 特殊晕染?? UpG_161515整 小晕染 调整阈值-10,
    # UpG_161533整 ok, UpG_161605整 ok 太浅那个得用轮廓,
    # UpG_161610整, UpG_161652整 那些红色的印子算不算晕染啊 mmp, UpG_161657整 ok, UpG_161701整 ok, UpG_161715整 ok,
    # UpG_161739 ok, UpG_161845墨团旁边, 还是误检出来一点宽度不宽
    # UpG_161908 字体印上去了 从这个标注可以看出,字体印不算晕染 但还是检测出来一点宽度,
    # UpG_161913乱)

    # 改进，不resize了
    # (UpG_161405 深色红水线 轮廓,UpG_161419. 深色红水线, UpG_161432 真奇怪框了一半 轮廓可以解决,UpG_161601 框多了 三角形晕染，
    #  UpG_161817, 也是深色红水线，UpG_161821 多框了点，UpG_161908 用轮廓可以解决, UpG_161908 字体印子)

    # ch #(UpG_161817, )
    # 1 正常晕染
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161519.jpg'  # 正常晕染
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161601.jpg'  # 轻微晕染

    # 2 粗线晕染
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161817.jpg'
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161903.jpg'
    # imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161913.jpg'


    # 3 正常粗1
    # imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092349.jpg'
    # 3 正常2
    # imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092623.jpg'


    # 最新变造票
    # imagePath = r'../img/redline/bloom_d_1.jpg'

    imagePath = r'E:\异常图0927\红水线晕染\94.jpg'
    img = cv_imread(imagePath)

    # 1.检测晕染，以及刮擦产生的晕染,ok的
    pairs = detect_bloom2(img, debug=True)

    # 2.检测白墨团，效果不错
    # detect_whiteink(img, debug=False)

    # 3.检测刮擦，刮出底色，基本是灰色
    # detect_scratch(img, debug=True)

    # 4.黑墨团检测
    # detect_blackink(img, debug=True)

    # bloom_test()

    cv2.imshow('imread', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

