
import xml.etree.ElementTree as ET
import numpy as np
import os
import cv2
from tools.xmlutils import get, get_and_check

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
    pass

if __name__ == '__main__':


    pass






