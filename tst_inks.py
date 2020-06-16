
from match_template import template_match

import xml.etree.ElementTree as ET
import numpy as np
import os

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union


if __name__ == '__main__':
    test_dir = r'E:\DataSet\hand_check_test\ch\UpG'
    anno_dir = r'E:\DataSet\hand_check_test\ch\UpG_anno'

    imgnames = os.listdir(test_dir)
    imgnames = [r'UpG_153614.bmp'] # UpG_153614.bmp UpG_153120.bmp
    imgpaths = [os.path.join(test_dir,img) for img in imgnames]


    categories = {'0': 0, '1': 1}
    INK_SUM = 0
    INK_FIND = 0

    RED_SUM = 0
    RED_FIND = 0

    for i in range(len(imgnames)):
        imgname = imgnames[i]
        imgpath = imgpaths[i]
        ids, locs, scores, isinks = template_match(imgpath, imgpath)

        inks_locs = np.array(locs[:-1])   # (18,4)
        redline_loc = np.array([locs[-1]])  # (1,4)

        # print(inks_locs.shape)
        # print(redline_loc.shape)

        print('test: ', imgpath)

        xmlname = imgname.split('.')[0] + '.xml'
        xml_f = os.path.join(anno_dir, xmlname)
        tree = ET.parse(xml_f)
        root = tree.getroot()

        inks_gt = []
        redline_gt = []

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text

            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)

            if categories[category] == 1:
                # redline
                redline_gt.append([xmin,ymin,xmax,ymax])
                continue
            inks_gt.append([xmin, ymin, xmax, ymax])

        inks_gt = np.array(inks_gt)
        redline_gt = np.array(redline_gt)

        # debug
        # print(inks_gt.shape)
        # print(redline_gt.shape)

        inks_ious = calc_iou(inks_gt,inks_locs)
        redline_iou = calc_iou(redline_gt,redline_loc)

        inks_maxious = np.max(inks_ious, axis=0)
        redline_maxiou = np.max(redline_iou, axis=0)

        print(inks_maxious)
        # print(redline_maxiou)


        f = len(np.where(inks_maxious>0.4)[0])
        s = len(inks_maxious)

        if f < s:
            print('\033[31;1m' + 'error miss: ' + imgpath + '\033[0m')

        INK_FIND += f
        INK_SUM += s

        if inks_maxious[0] > 0.5:
            RED_FIND += 1
        RED_SUM += 1

        # break

    print('%d : %d' % (INK_FIND, INK_SUM))
    print('%d : %d' % (RED_FIND, RED_SUM))
    # 1690: 1700
    # 100: 100

