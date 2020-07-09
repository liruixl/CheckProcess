
import xml.etree.ElementTree as ET
import os
import cv2
import random
from tools.xmlutils import get, get_and_check
from tools.imutils import cv_imread
from script.check100_split import  split


def split_irtr_noshuiyin():

    src_dir = r'E:\DataSet\荧光材质'
    dst_dir = r'E:\DataSet\IR\ir_no_zp'

    prefix = ['Upirtr']

    split(src_dir,dst_dir,prefix=prefix)



def crop_background():

    work_dir = r'E:\DataSet\IR\ir_no_zp\Upirtr'  # 21张不带背景的图像
    bg_dir = r'E:\DataSet\IR\zp_block\background'
    imgnames = os.listdir(work_dir)
    imgpaths = [os.path.join(work_dir, img) for img in imgnames]

    cnt = len(imgnames)

    # (min, max)
    block_y_w = (128, 141)
    block_y_h = (125, 176)

    block_zp_w = (205, 250)
    block_zp_h = (106, 180)


    idx = 1

    for i in range(cnt):
        print('process: ', imgpaths[i])
        ir_img = cv2.imread(imgpaths[i])  # 大约(300, 300)
        h, w, d = ir_img.shape


        # 每张图像裁剪 50 张 background
        # 2 * 25 ZP ￥ 各25张
        for _ in range(25):

            xmin = random.randint(10, w - 300)
            ymin = random.randint(10, h - 200)

            Y_W = random.randint(block_y_w[0], block_y_w[1])
            Y_H = random.randint(block_y_h[0], block_y_h[1])

            Y_IMG = ir_img[ymin : ymin + Y_H, xmin : xmin + Y_W]
            idx_str = str(idx).zfill(5)
            cv2.imwrite(os.path.join(bg_dir, idx_str + '.jpg'), Y_IMG)
            idx += 1

            xmin = random.randint(10, w - 300)
            ymin = random.randint(10, h - 200)

            ZP_W = random.randint(block_zp_w[0], block_zp_w[1])
            ZP_H = random.randint(block_zp_h[0], block_zp_h[1])

            ZP_IMG = ir_img[ymin : ymin + ZP_H, xmin : xmin + ZP_W]
            idx_str = str(idx).zfill(5)
            cv2.imwrite(os.path.join(bg_dir, idx_str + '.jpg'), ZP_IMG)
            idx += 1






def crop_zp():
    test_dir = r'E:\YHProject\票据处理\DataSet\IR_zengguang\img'
    anno_dir = r'E:\YHProject\票据处理\DataSet\IR_zengguang\anno'

    dst0_dir = r'E:\DataSet\IR\zp_block\0'
    dst1_dir = r'E:\DataSet\IR\zp_block\1'

    imgnames = os.listdir(test_dir)
    imgpaths = [os.path.join(test_dir, img) for img in imgnames]

    categories = {'0': 0, '1': 1}

    idx = 1
    # idx = str(idx)
    # fm = '0' * (5 - len(idx)) + idx

    for i in range(len(imgnames)):
        imgname = imgnames[i]
        imgpath = imgpaths[i]

        print('croping: ', imgpath)

        src_img = cv_imread(imgpath)

        xmlname = imgname.split('.')[0] + '.xml'
        xml_f = os.path.join(anno_dir, xmlname)

        if not os.path.isfile(xml_f):
            continue

        tree = ET.parse(xml_f)
        root = tree.getroot()

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text

            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)

            if xmax - xmin < 100 or ymax - ymin < 100:
                continue
            if category not in categories:
                continue

            if categories[category] == 0:
                dst_dir = dst0_dir
            else:
                dst_dir = dst1_dir

            dst_img = src_img[ymin:ymax, xmin:xmax]

            idx_str = str(idx).zfill(5)
            dst_f = os.path.join(dst_dir, idx_str + '.jpg')
            cv2.imwrite(dst_f, dst_img)

            idx += 1

if __name__ == '__main__':
    # imgpath = r'E:\DataSet\IR\zp_block\0\01552.jpg'
    # img = cv_imread(imgpath)
    # img = cv2.resize(img, (224,224))
    # cv2.imshow('res', img)
    # cv2.waitKey(0)

    crop_background()
