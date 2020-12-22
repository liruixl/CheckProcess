import cv2
import numpy as np
import erzhihua as erzhihua_util
#way_1
def way_1():
    img = cv2.imread('img/red_line.jpg')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])  # example value
    upper_red = np.array([10, 255, 255])  # example value
    # lower_red = np.array([165, 43, 46])  # example value
    # upper_red = np.array([180, 255, 255])  # example value
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    img_result = cv2.bitwise_and(img, img, mask=mask)
    cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
    cv2.imshow("binary2", img_result)


#way_3
def way_2():
    img = cv2.imread("img/UpG_redline/UpG_161305.jpg")
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower
    # mask(0 - 10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper
    # mask(170 - 180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # joed my mask
    mask = mask0 + mask1

    # 将我的输出img设置为零，除了我的掩码
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 255
    cv2.namedWindow("output_img", cv2.WINDOW_NORMAL)
    cv2.imshow("output_img", output_img)

    # 或你的HSV图像，我相信 * 是你想要的
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 255
    cv2.namedWindow("output_hsv", cv2.WINDOW_NORMAL)
    cv2.imshow("output_hsv", output_img)


def get_first(fushi):
    line_counts = 0
    flag = True
    img = fushi.copy()
    h = img.shape[0]
    w = img.shape[1]
    max_count = 0 #红水线长度
    for i in range(h):
        w_count = np.sum(img[i] == 0)
        max_count = max(max_count,w_count)
        if w_count > w * 3 / 7:
            img[i] = 0
            if flag:
                line_counts += 1
                flag = False
        else:
            img[i] = 255
            flag = True
    cv2.namedWindow("first", cv2.WINDOW_NORMAL)
    cv2.imshow('first', img)
    return img, line_counts, max_count

def get_second(fushi, img):
    h = img.shape[0]
    w = img.shape[1]
    count = 0
    len = w // 20
    for i in range(h):
        w_count = np.sum(img[i] == 0)
        if w_count > w * 2/3:
            for j in range(len, w, len):
                count = np.sum(fushi[i][j - len: j] == 0)
                if count >= len / 2:
                    img[i][j - len: j] = 0
                else:
                    img[i][j - len: j] = 255
        else:
            pass


    cv2.namedWindow("second", cv2.WINDOW_NORMAL)
    cv2.imshow('second', img)
    return img

def get_second2(fushi, img):
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

    cv2.namedWindow("second", cv2.WINDOW_NORMAL)
    cv2.imshow('second', img)
    return img

def test(img):
    binary_b2 = img
    h = binary_b2.shape[0]
    w = binary_b2.shape[1]
    count = 0
    len = w // 15
    for i in range(h):
        w_count = np.sum(binary_b2[i] == 0)
        if w_count > w * 1 / 2:
            for j in range(len, w, len):
                if binary_b2[i][j] == 0:
                    count = np.sum(binary_b2[i][j - len: j] == 0)  # 统计像素值小于50的像素点个数，50已经够宽泛了吧
                    if count >= len / 2:
                        binary_b2[i][j - len: j] = 0
                    else:
                        binary_b2[i][j - len: j] = 255
        else:
            binary_b2[i] = 255

    cv2.namedWindow("toutu", cv2.WINDOW_NORMAL)
    cv2.imshow('toutu', binary_b2)


def blacktowhite(img):
    img2 = img.copy()
    h = img2.shape[0]
    w = img2.shape[1]
    for i in range(h):
        for j in range(w):
            img2[i][j] = 255 - img2[i][j]
    cv2.namedWindow("blacktowhite", cv2.WINDOW_NORMAL)
    cv2.imshow('blacktowhite', img2)
    return img2

def test_danhua(ir_img):
    # gray = cv2.cvtColor(ir_img, cv2.COLOR_RGB2GRAY)
    binary = erzhihua_util.threshold_demo(ir_img)
    h = binary.shape[0]
    w = binary.shape[1]
    #按照50像素值来做判断
    len = 50
    is_danhua = False
    black_count = 0
    min_count = float("inf")
    max_count = 0
    for i in range(1, w, len):
        black_count = np.sum(binary[:,i:i+len] == 0)
        min_count = min(black_count, min_count)
        max_count = max(black_count, max_count)
        if black_count > min_count * 3 or black_count < max_count / 3:
            is_danhua = True
            # break
        if black_count < 100:
            break
        print(black_count)
        img_a = binary[:,i:i+len]
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow('test', img_a)
        cv2.waitKey(0)
    return is_danhua

def way_3():
    image = cv2.imread('img/red_line_new/red_5.png')
    # image = cv2.imread('img/UpG_redline/UpG_161314.jpg')
    # image = cv2.imread('img/read_line_5.png')
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.imshow('RGB', image)

    # binary = erzhihua_util.threshold_demo(image)
    #
    # h, w = image.shape[:2]
    # m = np.reshape(gray, [1, w * h])
    # mean = m.sum() / (w * h)
    # print("mean:",mean)
    # ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    # y_begin = 0
    # y_end = h - 1
    # x_begin = 0
    # x_end = w - 1

    # for i in range(h):
    #     count = np.sum(binary[i] == 0)
    #     if count < w:
    #         binary[i] = 255

    # cv2.namedWindow("new_img", cv2.WINDOW_NORMAL)
    # cv2.imshow('new_img', binary)

    b = image.copy()
    # set green and red channels to 0
    b[:, :, 1] = 0
    b[:, :, 2] = 0

    g = image.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    r = image.copy()
    # set blue and green channels to 0
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    # RGB - Blue
    # cv2.namedWindow("B-RGB", cv2.WINDOW_NORMAL)
    # cv2.imshow('B-RGB', b)

    # con_b = erzhihua_util.constract_brightness(b, 1.5, 1)
    # erzhihua_util.custom_threshold_old(r)

    # RGB - Green
    # cv2.namedWindow("G-RGB", cv2.WINDOW_NORMAL)
    # cv2.imshow('G-RGB', g)
    # con_g = erzhihua_util.constract_brightness(g, 1.5, 1)
    # erzhihua_util.custom_threshold_old(con_g)

    # erzhihua_util.threshold_demo(g)

    # RGB - Red
    # cv2.namedWindow("R-RGB", cv2.WINDOW_NORMAL)
    # cv2.imshow('R-RGB', r)
    # erzhihua_util.constract_brightness(r, 1.8, 1)

    is_danhua = test_danhua(image)
    con_b = erzhihua_util.constract_brightness(image, 1.0, 1)
    binary_b1 = erzhihua_util.threshold_demo(con_b)
    binary_b2 = erzhihua_util.local_threshold(con_b)
    binary_b2_fan =blacktowhite(binary_b2)
    # binary_b3_constract = erzhihua_util.constract_brightness(con_b, 1.8, 1)
    binary_b3 = erzhihua_util.custom_threshold_old(con_b)
    # pengzhang = erzhihua_util.dilate_demo(binary_b2)
    fushi = erzhihua_util.erode_demo(binary_b2)
    # pengzhang2 = erzhihua_util.dilate_demo(fushi)
    # ddenoo = erzhihua_util.denoosing(binary_b2, 3)
    img, line_counts, max_count = get_first(binary_b3)
    img = get_second2(binary_b3, img)
    print("红水线条数：", line_counts)
    print("红水线长度：", max_count) #630以内则为被破坏
    if is_danhua:
        print("红水线疑似淡化")



    cv2.waitKey(0)

if __name__ == '__main__':

    # way_1()
    # way_2()
    way_3()
    # image = cv2.imread('img/red_line_new/red_5.png')
    # test_danhua(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
