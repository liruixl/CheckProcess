#author:qx cheng time:2020/3/13
import cv2
import numpy as np
import os


def make_mask(img):
    h,w = img.shape[:2]
    mask = 255 - np.zeros_like(img)
    region = []
    if w < 1500:
        #为了去掉，红色logo、红水线区域，制作的的mask
        region.append([155,0,318,125])
        region.append([187,190,915,280])
        region.append([0,200,820,h])
    else:
        region.append([570,0,676,125])
        region.append([620,200,1280,280])
        region.append([0,0,430,h])
        region.append([430,200,1200,h])

    for cnt in region:
        mask[cnt[1]:cnt[3], cnt[0]:cnt[2]] = 0


    return mask

def sorce_hist(img, cnt):
    mask = np.zeros_like(img)
    x,y,w,h = cnt
    mask[ y:y+h,x:x+w ] = 255
    hist = cv2.calcHist([img], [0], mask, [2], [0, 256])
    histRatio = hist[1] / hist[0]
    # print(histRatio)
    # cv2.imshow('hist_mask', mask)
    return histRatio


def separate_color(img, color = 'red'):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == 'red':
        lower_hsv = np.array([165, 43, 46])  # 提取颜色的低值
        high_hsv = np.array([180, 255, 255])  # 提取颜色的高值
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
        lower_hsv = np.array([0, 43, 46])  # 提取颜色的低值
        high_hsv = np.array([10, 255, 255])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv))
    elif color == 'blue':
        lower_hsv = np.array([100,43,46])
        high_hsv = np.array([124,255,255])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    elif color == 'black':
        lower_hsv = np.array([0, 0, 0])
        high_hsv = np.array([180, 255, 46])



    cv2.imshow("mask", mask) # 白色为有校提取区域
    # get result
    res = cv2.bitwise_and(img, img, mask=mask) # 掩膜图像白色区域是对需要处理图像像素的保留

    work_dir = r'C:\Users\lirui\Desktop\temp\check\redline'
    temp_path = os.path.join(work_dir, 'redline_cqx.png')
    cv2.imwrite(temp_path, res)
    # cv2.namedWindow("maskColor", cv2.WINDOW_NORMAL)
    cv2.imshow("maskColor", res)

    return mask,res

def region_segmentation(img, color='red',seal=None):
    """
    :param img: The original image.
    :param color: Color channel extraction of img
    :param seal: seal location
    :return: the mask image with the color  and location in the picture.
    :maskColor: color channel picture from the original image
    :cnts: locations information
    """

    mask, maskColor = separate_color(img, color)

    if seal == None:

        gray = cv2.cvtColor(maskColor, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray1', gray)
        gray = cv2.bitwise_and(gray, make_mask(gray))
        # cv2.imshow('gray2', gray)
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rectKernel)  # 闭运算，膨胀，腐蚀
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, rectKernel)
        # cv2.imshow('closing', closing)
        _, thresh = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        ref, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = []

        for cnt in threshCnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            ar = w/float(h)
            if h > 20:
                if ar > 1.1 and ar < 20:
                    if w > 400 and h > 70:
                        cnts.append((x, y, w, int(h/2)))
                        cnts.append((x, y+int(h/2), w, int(h/2)))
                    else:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cnts.append((x,y,w,h))
        # for cnt in threshCnts:
        #     (x, y, w, h) = cv2.boundingRect(cnt)
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        #     cnts.append((x, y, w, h))
        print('附加字段')
        print(cnts)
        cv2.imshow('Additional word', img)

        return maskColor, cnts

    else:
        gray = cv2.cvtColor(maskColor, cv2.COLOR_BGR2GRAY)

        #检测圆形印章
        circles1 = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 1000
                                    , param1=100, param2=30, minRadius=100, maxRadius=200)

        if not(circles1 is None) :
            circles = circles1[0, :, :]
            circles = np.uint16(np.around(circles))
            for i in circles[:]:
                x, y, r = i[0] - i[2],i[1] - i[2],i[2]

                if sorce_hist(mask, [x,y,2*r,2*r]) > 0.25:
                    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
                    # cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 10)
                    cv2.rectangle(img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 2)

        # 检测方形印章
        canny = cv2.Canny(mask, 50,150)
        kernel = np.ones((5, 5), np.uint8)
        dilate = cv2.dilate(canny, kernel, iterations=1)


        contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]


        screenCnt = []
        # 遍历轮廓
        for c in contours:

            peri = cv2.arcLength(c, True)

            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 10000:
                screenCnt.append(approx)
        cv2.drawContours(img, screenCnt, -1, (0, 255, 0), 2)
        cv2.imshow("seal", img)

def get_location(img):
    mask, cnts = region_segmentation(img, color='red', seal = None)
    # region_segmentation(img,color = 'red',seal = 'seal')

def cv_read(in_path):
    cv_img = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), -1)  # -1表示cv2.IMREAD_UNCHANGED
    return cv_img

if __name__ == '__main__':
    work_dir = r'E:\DataSet\redline\UpG_redline'
    # work_dir = r'E:\DataSet\redline\UpG'

    temp_path = os.path.join(work_dir, 'UpG_161845.jpg')

    src = cv_read(temp_path)
    cv2.imshow('src', src)
    get_location(src)
    cv2.waitKey(0)