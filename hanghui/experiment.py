import cv2
import numpy as np


def find_tuanhua_roi():

    imFilename = r'img_tuanhua/tuanhua_uv_4.jpg'
    # imFilename = r'img_hanghui/zhongnong_uv.jpg'
    imFilename = r'img_dect/tuanhua_1.jpg'

    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # _, gray = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



    # 只寻找最外轮廓 RETR_EXTERNAL RETR_CCOMP
    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    H, W = gray.shape

    roi = np.zeros((H, W), dtype=np.int8)

    # 绘制轮廓
    cv2.drawContours(im, contours, -1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    cv2.drawContours(roi, contours, -1, 255, -1, lineType=cv2.LINE_AA)

    # 显示图像
    cv2.imshow('Contours', im)
    cv2.imshow('roi', roi)
    cv2.waitKey()
    cv2.destroyAllWindows()

    CONTOURS_1 = contours  # 保留一下，以防用到
    RECTS_1 = []
    AREA_1 = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        RECTS_1.append([x, y, x + w, y + h])
        AREA_1.append(area)

    cv2.imshow('?', gray)
    cv2.waitKey(0)

if __name__ == '__main__':
    find_tuanhua_roi()