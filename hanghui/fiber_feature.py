from tools.imutils import cv_imread, separate_color
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fiber_enclosing_circle(fiber_img, debug=False):
    im = fiber_img.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape
    # _, gray = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # 只寻找最外轮廓 RETR_EXTERNAL RETR_CCOMP
    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 1:
        # print("find {0} > 1 contours".format(len(contours)))
        return None, len(contours)

    # roi = np.ones((H, W), dtype=np.int8)
    # cv2.drawContours(roi, contours, -1, 255, -1, lineType=cv2.LINE_AA)
    # cv2.imshow('roi', roi)

    maxdist = 0

    center = (0, 0)
    for i in range(0, W):
        for j in range(0, H):
            dist = cv2.pointPolygonTest(contours[0],(i, j), True)
            if dist > maxdist:
                maxdist = dist
                center = (i, j)
    if debug:
        cv2.imshow('gray', gray)

        neiqieyuan = cv2.circle(im, center, int(maxdist), (0, 0, 255))
        cv2.namedWindow('circle', cv2.WINDOW_NORMAL)
        cv2.imshow('circle', im)
        cv2.waitKey(0)

    return center, maxdist

def tongji_oneimg_width():
    # tongji_width()

    ppp = r'E:\DataSet\fiber\fiber_img_1\00976.jpg'
    im = cv2.imread(ppp)

    center, r = fiber_enclosing_circle(im, debug=True)

def tongji_width():
    # 1 实验得出纤维丝的宽度长度
    # 2 二值化得到纤维丝的有效区域
    # 3 寻找轮廓
    # 4 寻找轮廓的最大内切圆


    imdir = r'E:\DataSet\fiber\f_blue'
    imdir = r'E:\DataSet\fiber\fiber_img_1'

    imnamelist = os.listdir(imdir)

    imnames = [os.path.join(imdir, f) for f in imnamelist]
    # imnames = imnames[:500]

    rs = []

    for impath in imnames:
        im = cv_imread(impath)
        center, radiesOrNum = fiber_enclosing_circle(im, False)
        if center is None:
            # print('{0} find {1} contours'.format(impath, radiesOrNum))
            continue

        if radiesOrNum > 4.5:
            print(impath, ' R>4.5, feel exption')
        rs.append(radiesOrNum * 2)

    xs = [i for i in range(len(rs))]
    plt.plot(xs, rs, "ob")
    plt.axis([0, 800, 0, 5])
    plt.show()


def fiber_hsv():
    impath = r'E:\DataSet\fiber\fiber_img_1\00354.jpg'

    imdir = r'E:\DataSet\fiber\f_blue'  # 保存明显blue的img文件夹
    imnamelist = os.listdir(imdir)
    imnames = [os.path.join(imdir, f) for f in imnamelist]

    HC = []
    SC = []
    VC = []

    for impath in imnames:

        im = cv_imread(impath)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # _, gray = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 1:
            # print("find {0} > 1 contours".format(len(contours)))
            print("{0} find {1} > 1 contours".format(impath, len(contours)))
            continue
        else:
            # print("{0} find {1} == 1 contours".format(impath, len(contours)))
            pass

        # roi = np.zeros((H, W), dtype=np.int8)
        # cv2.drawContours(roi, contours, -1, 255, -1, lineType=cv2.LINE_AA)
        # cv2.imshow('roi', roi)
        # cv2.waitKey(0)

        for i in range(0, H):
            for j in range(0, W):
                is_in = cv2.pointPolygonTest(contours[0], (i, j), False)

                if is_in >= 0:
                    HC.append(hsv[i][j][0])
                    SC.append(hsv[i][j][1])
                    VC.append(hsv[i][j][2])

    plt.hist(HC, 180)
    plt.show()
    plt.hist(SC, 256)
    plt.show()
    plt.hist(VC, 256)
    plt.show()


if __name__ == '__main__':
    # fiber_hsv()
    # name = '00354.jpg'  # 明显蓝 ok
    # name = '00647.jpg'  # 浅蓝 ok
    # name = '00619.jpg'  # 浅蓝 false
    #
    # name = '00618.jpg'  # 浅红 false
    # name = '00714.jpg'  # 浅红 ok
    ppp = r'E:\DataSet\fiber\fiber_img_1\00168.jpg'
    im = cv2.imread(ppp)
    mask, res = separate_color(im, 'bright_blue')

    # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow('mask', mask)
    cv2.imshow('res', im)

    cv2.waitKey(0)

