import cv2
import numpy as np
import matplotlib.pyplot as plt


# normal
# bank_icon = cv2.imread(r'../img_hanghui/zhongnong_uv.jpg')

# blist = [r'img/normal_1.jpg', r'img/normal_2.jpg', r'img/normal_3.jpg', r'img/normal_4.jpg', r'img/normal_5.jpg']

# DIY
# blist = [r'diy/normal_1.jpg', r'diy/normal_2.jpg', r'diy/normal_3.jpg', r'diy/normal_3 (2).jpg', r'diy/normal_4.jpg', r'diy/normal_5.jpg']

blist = [r'img/normal_11.jpg', r'img/normal_12.jpg', r'img/normal_13.jpg', r'img/normal_14.jpg', r'img/normal_15.jpg']

# blist = [r'img/color_1.jpg', r'img/color_2.jpg', r'img/color_3.jpg', r'img/color_4.jpg']

# blist = [r'img/color_4.jpg']

# not normal

# bank_icon = cv2.imread(r'img/normal_1.jpg')

# bank_icon = cv2.imread(r'img/color_2.jpg')


for pt in blist:
    bank_icon = cv2.imread(pt)

    bank_gray = cv2.cvtColor(bank_icon, cv2.COLOR_BGR2GRAY)

    _, bank_bin = cv2.threshold(bank_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # bank_bin = cv2.erode(bank_bin, element1, iterations=1)


    bank_only = cv2.bitwise_and(bank_icon, bank_icon, mask=bank_bin)
    bank_gray_only = cv2.bitwise_and(bank_gray, bank_gray, mask=bank_bin)

    b = bank_only[:,:,0]
    g = bank_only[:,:,1]
    r = bank_only[:,:,2]


    b = b.ravel()  # (1, n)
    b = [n for n in b if n != 0]

    g = g.ravel()  # (1, n)
    g = [n for n in g if n != 0]

    r = r.ravel()  # (1, n)
    r = [n for n in r if n != 0]

    gy = bank_gray_only.ravel()
    gy = [n for n in gy if n != 0]

    gy_mean = np.mean(gy)
    gy_var = np.var(gy, ddof=1)
    print('均值和方差分别为', gy_mean, gy_var)


    # 均值方差

    # plt.hist(b, 256)
    # plt.show()
    # plt.hist(g, 256)
    # plt.show()
    # plt.hist(r, 256)
    # plt.show()

    plt.hist(gy, 256)
    plt.show()

    # cv2.imshow('img', np.hstack([bank_icon, bank_only]))
    # cv2.imshow('bank_gray', np.hstack([bank_gray, bank_bin]))
    # cv2.waitKey(0)