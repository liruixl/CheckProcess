import cv2 as cv
import numpy as np


list = [[(82, 125),(271, 150)],
[(411, 125),(437, 150)],
[(541, 125),(561, 149)],
[(660, 126),(678, 149)],
[(760, 124),(889, 150)],
[(910, 124),(1039, 150)],
[(82, 162),(164, 189)],
[(762, 165),(889, 190)],
[(86, 310),(135, 335)],
[(85, 354),(230, 382)],
[(84, 400),(230, 430)],
[(84, 450),(206, 478)],
[(759, 307),(810, 334)],
[(762, 356),(811, 381)],
[(763, 450),(812, 475)],
[(924, 451),(971, 476)],
[(99, 212),(178, 268)],
[(891, 204),(1226, 235)],
[(27, 170),(53, 469)]]


def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    # print("threshold value %s"%ret)
    # cv.namedWindow("binary0", cv.WINDOW_NORMAL)
    # cv.imshow("binary0", binary)
    return binary

def custom_threshold_old(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    # print("mean:",mean)
    ret, binary =  cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    # cv.namedWindow("binary2", cv.WINDOW_NORMAL)
    # cv.imshow("binary2", binary)
    return binary

def constract_brightness(src1, a, g):
    src2 = np.zeros(src1.shape, src1.dtype)
    dst = cv.addWeighted(src1, a, src2, 1-a, g)
    # cv.namedWindow("constract", cv.WINDOW_NORMAL)
    # cv.imshow("constract", dst)
    return dst

def black(img):

    # custom_threshol    d_old(img)

    h = img.shape[0]
    w = img.shape[1]
    count = 0
    for i in range(h):
        count += np.sum(img[i] < 50) #统计像素值小于50的像素点个数，50已经够宽泛了吧
    return count



def isExist(origin_img, img,  left_up, right_down):
    # origin_img = threshold_demo(origin_img)
    # img = constract_brightness(img, 1.8, 1)
    # img = custom_threshold_old(img)
    x1,y1 = left_up
    x2,y2 = right_down
    new_origin_img = origin_img[y1:y2, x1:x2]
    new_img = img[y1:y2, x1:x2]
    origin_img_black_count = black(new_origin_img)
    img_count = black(new_img)
    result = 1 - img_count/origin_img_black_count
    # print('红外保护油墨置信度：', result);
    # if count > 30: #若像素点为黑色大于30个
    #     print("这个框不是红外保护油墨")
    # else:
    #     print("是红外保护油墨")
    return result




if __name__ == '__main__':
    origin_img = cv.imread('img/pufa_rgb.jpg')
    img = cv.imread('img/Upir_3.jpg')
    # img = constract_brightness(img, 1.8, 1)
    # img = custom_threshold_old(img)
    # isExist(img,(910, 124),(1039, 150))
    for i in range(len(list)):
        isExist(origin_img, img, list[i][0], list[i][1])
    cv.waitKey(0)
    cv.destroyAllWindows()

