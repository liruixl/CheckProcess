import cv2
import numpy as np
import os
from tools.imutils import cv_imread
from red_waterline.text_detection import detect_bloom2

def get_first(fushi):
    img = fushi.copy()
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        w_count = np.sum(img[i] == 0)
        if w_count > w * 3 / 7:
            img[i] = 0
        else:
            img[i] = 255
    # cv2.namedWindow("first", cv2.WINDOW_NORMAL)
    # cv2.imshow('first', img)
    return img

def get_second2(fushi, image):
    img = image.copy()
    h = img.shape[0]
    w = img.shape[1]
    count = 0
    len = 10
    for i in range(h):
        w_count = np.sum(img[i] == 0)
        if w_count > w * 2/3:
            for j in range(len, w-len, 1):
                # if fushi[i][j] == 0:
                count_left = np.sum(fushi[i][j - len: j] == 0)  # 统计像素值小于50的像素点个数，50已经够宽泛了吧
                count_right = np.sum(fushi[i][j : j + len] == 0)
                if count_left <= len // 2 and count_right <= len // 2:
                    img[i][j - len: j + len] = 255
        else:
            pass

    for i in range(h):
        black_count = np.sum(img[i] == 0);
        if black_count > 0:
            for j in range(w):
                if(img[i][j] == 0):
                    if i-1 >= 0 and 0 in img[i-1]:
                        img[i-1][j] = 0
                    if i+1 < h and 0 in img[i+1]:
                        img[i+1][j] = 0

    for i in range(h):
        black_count = np.sum(img[i] == 0);
        if black_count > 0:
            for j in range(w):
                if(img[i][j] == 0):
                    if i-1 >= 0 and 0 in img[i-1]:
                        img[i-1][j] = 0
                    if i+1 < h and 0 in img[i+1]:
                        img[i+1][j] = 0

    # cv2.namedWindow("second", cv2.WINDOW_NORMAL)
    # cv2.imshow('second', img)
    return img


def constract_brightness(src1, a, g):
    src2 = np.zeros(src1.shape, src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)
    # cv2.namedWindow("constract", cv2.WINDOW_NORMAL)
    # cv2.imshow("constract", dst)
    return dst

def custom_threshold_old(image):
    # gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    # print("mean:",mean)
    ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
    # cv2.imshow("binary2", binary)
    return binary


def confidence_percentage(org_img, img):
    h = img.shape[0]
    w = img.shape[1]
    org_count = 0
    img_count = 0

    #统计原图上黑色像素数量（原来红水线像素个数）
    for i in range(h):
        org_count += np.sum(org_img[i] == 0)

    #统计被破坏后红水线像素剩余个数
    for i in range(h):
        img_count += np.sum(img[i] == 0)


    #统计剩余的百分比
    red_percentage = img_count / org_count

    #统计被破坏的百分比
    result  = 1 - red_percentage

    print("被破坏的百分比为{}%".format(round(result * 100, 2)))

    return result


def way_3(image, debug=False):

    # image = cv2.imread(img_path)
    # cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    # cv2.imshow('RGB', image)
    con_b = constract_brightness(image, 1.5, 1)
    binary_b3 = custom_threshold_old(con_b)
    first = get_first(binary_b3)
    img = get_second2(binary_b3, first)

    corrs = detect_bloom2(image, debug=debug)

    for xmin,ymin,xmax,ymax in corrs:
        img[ymin:ymax, xmin:xmax] = 255
    if debug:
        cv2.namedWindow("second", cv2.WINDOW_NORMAL)
        cv2.imshow('second', img)

    damage = confidence_percentage(first, img)

    # debug = np.vstack((image,img))
    # cv2.imshow('res', debug)

    return damage

def debug():
    check_dir = r'E:\DataSet\redline\UpG_redline'
    debug_dir = r'E:\DataSet\redline\debug'

    filename_list = os.listdir(check_dir)
    for imgname in filename_list:
        img_path = os.path.join(check_dir, imgname)
        debug = way_3(img_path)
        cv2.imwrite(os.path.join(debug_dir, imgname), debug)

if __name__ == '__main__':


    img_path = r'E:\DataSet\redline\UpG_redline\UpG_161436.jpg'
    way_3(img_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()