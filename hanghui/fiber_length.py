from tools.imutils import cv_imread, separate_color
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def fiber_length(fiber_img, debug=False):
    im = fiber_img.copy()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape
    # _, gray = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # 只寻找最外轮廓 RETR_EXTERNAL RETR_CCOMP
    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 1:
        # print("find {0} > 1 contours".format(len(contours)))
        return 0

    if debug:
        roi = fiber_img.copy()
        cv2.drawContours(roi, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        # cv2.imshow('bin', gray)
        # cv2.imshow('ori', im)
        # cv2.imshow('roi', roi)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        cv2.imshow('res', np.hstack((im,gray,roi)))
        cv2.waitKey()

    fiber_lenght = cv2.arcLength(contours[0], True)

    return fiber_lenght

def tongji_oneimg_length():
    # tongji_width()

    warning_imgname = ['00014.jpg', '00848.jpg', '00890.jpg', '00890.jpg', '00964.jpg', '01209.jpg', \
                       '01321.jpg', '01341.jpg']
    img_dir = r'E:\DataSet\fiber\fiber_img_1'
    for name in warning_imgname:

        ppp = os.path.join(img_dir, name)
        im = cv2.imread(ppp)

        length = fiber_length(im, debug=True)
        print(name, "length", length)

def tongji_length():
    # 1 实验得出纤维丝的宽度长度
    # 2 二值化得到纤维丝的有效区域
    # 3 寻找轮廓
    # 4 寻找轮廓的最大内切圆, 轮廓长度


    imdir = r'E:\DataSet\fiber\f_blue'
    imdir = r'E:\DataSet\fiber\fiber_img_1'

    imnamelist = os.listdir(imdir)

    imnames = [os.path.join(imdir, f) for f in imnamelist]
    # imnames = imnames[:500]

    rs = []

    for impath in imnames:
        im = cv_imread(impath)
        length = fiber_length(im, False)
        if length == 0:
            # print('{0} find {1} contours'.format(impath, radiesOrNum))
            continue

        if length//2 > 80:
            print('[WARNING] {0} find {1} length contours'.format(impath, length//2))

        rs.append(length / 2)

    xs = [i for i in range(len(rs))]
    plt.plot(xs, rs, "ob")
    plt.axis([0, 800, 0, 100])
    plt.show()



if __name__ == '__main__':
    # fiber_hsv()
    # name = '00354.jpg'  # 明显蓝 ok
    # name = '00647.jpg'  # 浅蓝 ok
    # name = '00619.jpg'  # 浅蓝 false
    #
    # name = '00618.jpg'  # 浅红 false
    # name = '00714.jpg'  # 浅红 ok


    # 分离颜色可视化
    # ppp = r'E:\DataSet\fiber\fiber_img_1\01224.jpg'
    # im = cv2.imread(ppp)
    # mask, res = separate_color(im, 'bright_blue')
    #
    # # cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', im)
    #
    # cv2.waitKey(0)

    # tongji_length()
    tongji_oneimg_length()



