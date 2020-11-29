
import cv2
import numpy as np


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


    cv2.imshow('preprocess', np.vstack([sobel, binary, erosion, dilation2]))
    cv2.waitKey(0)

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
    cv2.waitKey(0)
    return xs, ys, ws, hs


if __name__ == '__main__':
    upir = r'img/upir_1.jpg'
    upir_img = cv2.imread(upir)
    gray = cv2.cvtColor(upir_img, cv2.COLOR_BGR2GRAY)
    dila = preprocess(gray)
    findTextRegion(upir_img, dila)
