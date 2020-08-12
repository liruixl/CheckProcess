import cv2
import numpy as np
import os
from tools.imutils import cv_imread

check_scale = (1330, 630)
redline_scale = (666, 66)  # 66666666 temp(56, 79)


def split_redline(upg_img_path, dst_dir, method=cv2.TM_CCOEFF_NORMED):

    target = cv_imread(upg_img_path)
    cropx_offset = 0

    ori_h, ori_w = target.shape[:2]
    if ori_w > check_scale[0]:
        cropx_offset = ori_w - check_scale[0]
        target = target[:, cropx_offset:]  # 裁剪坐标为[y0:y1, x0:x1]

    print(target.shape)

    # 1 取人名币大写的模板，用于定位红水线
    temp = cv2.imread(r'../img/money_chinese.jpg')
    t_height, t_width = temp.shape[:2]
    print(temp.shape[:2])

    # 2 划可能的区域
    # (1330, 630) [xmin,ymin,xmax,ymax]
    left_up = [60, 100, 400, 320]  # 左上区域
    possible_area = left_up
    xmin, ymin, xmax, ymax = possible_area
    target_crop = target[ymin:ymax, xmin:xmax]

    result = cv2.matchTemplate(target_crop, temp, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        t_left = min_loc
    else:
        t_left = max_loc  # (x,y) 访问 result[y][x], row col

    # 3 将坐标映射回原图
    tl = (t_left[0]+xmin, t_left[1]+ymin)
    br = (tl[0]+t_width, tl[1]+t_height)

    top_right = (br[0], tl[1])

    xmin, ymin  = redline_tl = (top_right[0] + 4, top_right[1] - 6)
    xmax, ymax  = redline_br = (redline_tl[0] + 666, redline_tl[1] + 66)

    redline_img = target[ymin : ymax, xmin:xmax]
    filename = upg_img_path.split('\\')[-1]
    filename = filename.split('.')[0]
    cv2.imwrite(os.path.join(dst_dir, filename) + '.jpg', redline_img, [int(cv2.IMWRITE_JPEG_QUALITY),95])

    # cv2.rectangle(target, tl, br, (0,0,255), 1)
    # cv2.rectangle(target, redline_tl, redline_br, (255,0,0), 1)
    # cv2.imshow('moneychinese', target)


if __name__ == '__main__':
    dst_dir = r'E:\DataSet\redline_ok\redline_normal'

    check_dir = r'E:\DataSet\redline_ok\UpG'

    filenames = os.listdir(check_dir)

    for imgname in filenames:
        split_redline(os.path.join(check_dir, imgname), dst_dir)

    # cv2.waitKey(0)


