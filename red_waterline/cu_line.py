
import cv2
import numpy as np

from tools.imutils import cv_imread,separate_color
from red_waterline.union_find import UnionFind


G = 88
TH = 28

# 1 正常晕染
imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161405.jpg'
# 2 粗线晕染
# imagePath = r'E:\DataSet\redline\UpG_redline\UpG_161817.jpg'
# 3 正常1
# imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092349.jpg'
# 3 正常2
# imagePath = r'E:\DataSet\redline_ok\redline_normal\UpG_092610.jpg'
img = cv_imread(imagePath)


def detect_darkline(img, debug=False):

    mask_red, redhsv = separate_color(img, color='red')

    h, w, _ = img.shape
    for i in range(0, h):
        for j in range(0, w):
            a, b, c = redhsv[i][j]
            if a == 0 and b == 0 and c==0:
                redhsv[i][j] = [255,255,255]

    # cv2.imshow('ori', img)
    # cv2.imshow('tiqutu', redhsv)
    # cv2.imshow('baidi', redhsv)
    if debug:
        cv2.imshow('tiqu', np.vstack((img,redhsv)))

    res_gray = cv2.cvtColor(redhsv, cv2.COLOR_BGR2GRAY)
    print(res_gray.shape)
    vals = []
    for i in range(0, h):
        for j in range(0, w):
            if res_gray[i][j] != 255:
                vals.append(res_gray[i][j])
    vals = np.array(vals)

    # plt.hist(vals,255)
    # plt.show()

    line_gray_mean = int(np.mean(vals))
    print('红水线灰度化均值:', line_gray_mean)

    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.createTrackbar('G', 'image', 88, 255, nothing)
    # cv2.createTrackbar('TH', 'image', 28, 40, nothing)
    # G = cv2.getTrackbarPos('G', 'image')
    # TH = cv2.getTrackbarPos('TH', 'image')


    # G 与 红水线均值有关, 均值颜色越深，我们的阈值也应该越深
    # G = 90
    # G = int(line_gray_mean)*2 - 110

    cha = -line_gray_mean + 110
    G = line_gray_mean -  (10 if cha<10 else cha)

    print('阈值为：', G)
    _, res = cv2.threshold(res_gray,G,255,cv2.THRESH_BINARY)

    ser = cv2.bitwise_not(res)

    # 以(30,2)长度的滑动窗口，滑动，如果有校像素超过3/4 28? ,认为是深色红水线
    # 中值滤波呢，不太行
    # midblur = cv2.medianBlur(ser, 3)

    valid = []
    blank = np.zeros((h, w), dtype=np.uint8)  # 黑板

    kh = 2
    kw = 20

    # TH 经验值
    for i in range(0,h-kh):
        for j in range(0,w-kw):
            temp = ser[i:i+kh, j:j+kw]
            white = np.sum(temp==255)
            if white >= TH:
                blank[i:i + kh, j:j + kw] = 255

    if debug:
        cv2.imshow('gray th' + str(G), np.vstack((res_gray,res,ser)))

    _, contours, hierarchy = cv2.findContours(blank, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print('find conrours:', len(contours))
    cu_lines = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        # area = cv2.contourArea(cnt)
        cu_lines.append([x,y,x+w,y+h])

    print('find lines:',len(cu_lines))
    n_lines = len(cu_lines)
    uf = UnionFind(n_lines)

    for i in range(0,n_lines - 1):
        for j in range(i + 1, n_lines):
            xmin1, ymin1, xmax1,ymax1 = cu_lines[i]
            xmin2, ymin2, xmax2,ymax2 = cu_lines[j]

            w1, h1 = xmax1 - xmin1, ymax1-ymin1
            w2, h2 = xmax2 - xmin2, ymax2-ymin1

            cx1,cy1 = xmin1 + w1//2, ymin1 + h1//2
            cx2,cy2 = xmin2 + w2//2, ymin2 + h2//2

            x_low = max(xmin1,xmin2)
            x_high = min(xmax1,xmax2)

            if abs(cy1 - cy2) > 10:
                continue

            if x_high - x_low <= 0:
                continue

            iou = x_high-x_low
            percent_1 = iou/(xmax1 - xmin1)
            percent_2 = iou/(xmax2 - xmin2)

            if percent_1 > 0.5 and percent_2 > 0.5:
                uf.union(i,j)

    # 并查集合并结束
    D = {}
    for idx in range(0, n_lines):
        p = uf.find(idx)
        if p in D.keys():
            D[p].append(idx)
        else:
            D[p] = []
            D[p].append(p)

    print('并查集结果：',D)
    rect_merge = []
    for k, rects in D.items():
        if (len(rects) == 1):
            rect_merge.append(cu_lines[rects[0]])
            xmin, ymin, xmax, ymax = cu_lines[rects[0]]
        else:
            xmin = np.mean([cu_lines[r][0] for r in rects])
            ymin = min([cu_lines[r][1] for r in rects])
            xmax = np.mean([cu_lines[r][2] for r in rects])
            ymax = max([cu_lines[r][3] for r in rects])

            rect_merge.append([int(xmin), ymin, int(xmax), ymax])

    culine_merge_rect = []

    img_bgr = img.copy() # 用于绘制
    blank_bgr = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR) # 用于绘制必须是BGR图
    for r in rect_merge:
        xmin, ymin, xmax, ymax = r
        if ymax - ymin < 10:
            cv2.rectangle(blank_bgr, (xmin, ymin), (xmax, ymax), (255, 200, 0), 1)
        else:
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.rectangle(blank_bgr, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            culine_merge_rect.append(r)
    if debug:
        cv2.imshow('line merge', blank_bgr)

    # return culine_merge_rect

    return np.vstack((img_bgr,blank_bgr))


if __name__ == '__main__':
    detect_darkline(img, debug=True)
    cv2.waitKey(0)