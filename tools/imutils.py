import numpy as np
import cv2
from collections import Counter

# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR) # 有疑问
    return cv_img

# 最大类间方差确定阈值 输入灰度图
def _var_max_interclass(img, calc_zero=True):
    h, w = img.shape[:2]
    mean = np.mean(img)

    counts = np.zeros((256,), dtype=int)
    prob = np.zeros((256,), dtype=float)

    count_map = Counter(img.ravel())
    for pix_val, count in count_map.items():
        if not calc_zero and pix_val == 0:
            continue
        counts[pix_val] = count
    prob = counts / (h*w)

    ks = [i for i in range(90,160)]
    maxvar = 0
    threshold = 0

    for k in ks:
        p1 = np.sum(prob[:k])
        p2 = 1-p1

        mean1 = [i*prob[i] for i in range(k)]
        # mean1 = np.sum(mean1)/p1
        mean1 = np.sum(mean1)


        # mean11 = [i * counts[i] for i in range(k)]
        # sum11 = np.sum(counts[:k])
        # mean11 = np.sum(mean11) / sum11

        var = (mean*p1 - mean1)**2 / (p1*p2)

        if var > maxvar:
            maxvar = var
            threshold = k

    return threshold


def separate_color(img, color = 'red'):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # H: （100，130）S: （75，225） V: （50，255) 背景未被去掉
    if color == 'bright_blue':
        # lower_hsv = np.array([100, 75, 50]) # 背景并没有过滤阈值太大
        # high_hsv = np.array([130, 225, 255])

        # lower_hsv = np.array([110, 125, 70])  # 怎么提的都是背景了，难道背景多?
        # high_hsv = np.array([120, 175, 100])

        lower_hsv = np.array([100, 70, 130])  # 单独截取 亮蓝 部分
        high_hsv = np.array([120, 190, 255])

        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    elif color == 'red':
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
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    elif color == 'white':
        lower_hsv = np.array([0, 0, 180])  # (0,0,221) V 180 正合适
        high_hsv = np.array([180, 30, 255])  # 调整S看不到效果
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    elif color == 'gray':
        lower_hsv = np.array([0, 0, 46])
        high_hsv = np.array([180, 43, 220])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    elif color == 'green':
        lower_hsv = np.array([35, 43, 46])
        high_hsv = np.array([99, 255, 255])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    elif color == 'gratch':
        lower_hsv = np.array([0, 0, 140])  # H [0,30] and [160,180]
        high_hsv = np.array([30, 20, 180]) # S [0,20]最好
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)

        lower_hsv = np.array([160, 0, 140])  #
        high_hsv = np.array([180, 20, 180])  # V [140,180]宽松, [150,170]严格
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv))
    elif color == 'yingzhang':
        # H([0,10] and [175,180]) S[160/180,230]  V [70/75,90/95]
        lower_hsv = np.array([0, 160, 70])
        high_hsv = np.array([10, 230, 95])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)

        lower_hsv = np.array([175, 160, 70])
        high_hsv = np.array([180, 230, 95])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv))
    elif color == 'dark_red':
        lower_hsv = np.array([170, 60, 70])
        high_hsv = np.array([180, 115, 120])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)

    # cv2.imshow("mask", mask) # 白色为有校提取区域
    # get result
    res = cv2.bitwise_and(img, img, mask=mask)

    return mask,res


def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
