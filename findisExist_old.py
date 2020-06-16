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

def custom_threshold_old(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    # print("mean:",mean)
    ret, binary =  cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    return binary

def constract_brightness(src1, a, g):
    src2 = np.zeros(src1.shape, src1.dtype)
    dst = cv.addWeighted(src1, a, src2, 1-a, g)
    # cv.namedWindow("constract", cv.WINDOW_NORMAL)
    # cv.imshow("constract", dst)
    return dst

def black(img):

    h = img.shape[0]
    w = img.shape[1]
    count = 0
    for i in range(h):
        count += np.sum(img[i] < 50) #统计像素值小于50的像素点个数，50已经够宽泛了吧
    return count


def is_prot_ink(upir_img, left_up, right_down):
    x1,y1 = left_up
    x2,y2 = right_down

    new_img = upir_img[y1:y2, x1:x2]
    count = black(new_img)
    if count > 30:  # 若像素点为黑色大于30个
        print('count black : {0}'.format(count))
        return False
    else:
        return True


if __name__ == '__main__':
    img = cv.imread( r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\check_UpG_Upir\092342_Upir.jpg')
    img = constract_brightness(img, 1.8, 1)
    img = custom_threshold_old(img)
    for i in range(len(list)):
        is_prot_ink(img, (149, 124), (338, 149))

