
import xml.etree.ElementTree as ET
import numpy as np
import random
import os
import cv2
from tools.xmlutils import get, get_and_check
from tools.bbox_utils import calc_iou

def crop_fibers():
    test_dir = r'E:\DataSet\hand_check_test\Upuv_test\images'
    anno_dir = r'E:\DataSet\hand_check_test\Upuv_test\annotations'

    dst_dir = r'E:\DataSet\fiber\fiber_img_2'

    imgnames = os.listdir(test_dir)
    imgpaths = [os.path.join(test_dir, img) for img in imgnames]

    categories = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

    idx = 1
    # idx = str(idx)
    # fm = '0' * (5 - len(idx)) + idx

    for i in range(len(imgnames)):
        imgname = imgnames[i]
        imgpath = imgpaths[i]

        print('croping: ', imgpath)

        src_img = cv2.imread(imgpath)

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

            if xmax - xmin > 70 or ymax - ymin > 70:
                continue

            if categories[category] == 4:
                dst_img = src_img[ymin:ymax, xmin:xmax]

                idx_str = str(idx).zfill(5)
                dst_f = os.path.join(dst_dir, idx_str + '.jpg')
                cv2.imwrite(dst_f, dst_img)

                idx += 1


def crop_background():
    test_dir = r'E:\DataSet\CheckData_UV\images'
    anno_dir = r'E:\DataSet\CheckData_UV\Annotations_voc'

    dst_dir = r'E:\DataSet\CheckData_UV\background_block'

    imgnames = os.listdir(test_dir)
    imgpaths = [os.path.join(test_dir, img) for img in imgnames]

    categories = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

    idx = 1


    for i in range(len(imgnames)):
        imgname = imgnames[i]
        imgpath = imgpaths[i]

        print('croping: ', imgpath)

        src_img = cv2.imread(imgpath)

        xmlname = imgname.split('.')[0] + '.xml'
        xml_f = os.path.join(anno_dir, xmlname)

        if not os.path.isfile(xml_f):
            continue

        tree = ET.parse(xml_f)
        root = tree.getroot()

        # 统计纤维丝的框，保证随机裁剪的背景不是纤维丝
        fiber_bboxs = []
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text

            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)

            if xmax - xmin > 70 or ymax - ymin > 70:
                continue

            if categories[category] == 4:
                fiber_bboxs.append([xmin,ymin,xmax,ymax])

        # 每张图裁剪cnt个背景框
        src_h, src_w, _ = src_img.shape
        dst_h, dst_w = 50, 50
        bbox2 = np.array(fiber_bboxs)
        cnt = 15
        for i in range(cnt):

            # 随机生成裁剪坐标
            xmin = 0
            ymin = 0
            j = 10  # 尝试10次随机
            while True and j >= 0:
                j -= 1
                xmin = random.randint(100, src_w - dst_w - 2)
                ymin = random.randint(2, src_h - dst_h - 2)

                bbox1 = np.array([xmin, ymin, xmin+dst_w, ymin+dst_h])
                ious = calc_iou(bbox1,bbox2)  # (1, n)
                iou_max = np.max(ious)
                if iou_max < 0.2:
                    break

            dst_img = src_img[ymin:ymin+dst_h, xmin:xmin+dst_w]

            idx_str = str(idx).zfill(5)
            dst_f = os.path.join(dst_dir, idx_str + '.jpg')
            cv2.imwrite(dst_f, dst_img)

            idx += 1

        # break


if __name__ == '__main__':

    crop_background()






