import cv2
import os
import numpy as np
import random
from tools.xmlutils import get, get_and_check
import xml.etree.ElementTree as ET

from tools.imutils import rotate, translate


def crop_block():
    anno_dir = r'E:\DataSet\yingguang_new\yingguang_real_alone'
    img_dir = r'E:\DataSet\yingguang_new\yingguang_new_alone'
    dst_dir = r'E:\DataSet\yingguang_new\yg_block'  # 18

    idx = 1

    anno_list = os.listdir(anno_dir)
    imgnames = [anno.split('.')[0] + '.bmp' for anno in anno_list]
    imgpaths = [os.path.join(img_dir, n) for n in imgnames]

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

            if category == '5':
                dst_img = src_img[ymin:ymax, xmin:xmax]

                idx_str = str(idx).zfill(5)
                dst_f = os.path.join(dst_dir, idx_str + '.jpg')
                cv2.imwrite(dst_f, dst_img)

                idx += 1

# N(N-1)/2 * 2 形变翻转
def merge_block():

    yg_block_dir = r'E:\DataSet\yingguang_new\yg_block'
    yg_block_merge = r'E:\DataSet\yingguang_new\yg_block_merge'

    imgnames = os.listdir(yg_block_dir)
    imgpaths = [os.path.join(yg_block_dir, img) for img in imgnames]

    cnt = len(imgnames)

    for i in range(cnt):
        for j in range(i + 1, cnt):
            img1 = cv2.imread(imgpaths[i])
            img2 = cv2.imread(imgpaths[j])
            h1, w1, _ = img1.shape
            h2, w2, _ = img2.shape

            h = (h1 + h2) // 2
            w = (w1 + w2) // 2

            img1 = cv2.resize(img1, (h, w))
            img2 = cv2.resize(img2, (h, w))

            res = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

            m = str(i) + '_' + str(j)
            cv2.imwrite(os.path.join(yg_block_merge, m + '.jpg'), res)

            rhw = max(h, w)
            res = cv2.resize(res, (rhw, rhw))
            res = rotate(res, 90)

            res_r1 = cv2.resize(res, (h, w))
            r1 = m + '_r1'
            # cv2.imwrite(os.path.join(yg_block_merge, r1 + '.jpg'), res_r1)

            res = rotate(res, 90)
            res_r2 = cv2.resize(res, (h, w))
            r2 = m + '_r2'
            cv2.imwrite(os.path.join(yg_block_merge, r2 + '.jpg'), res_r2)

            res = rotate(res, 90)
            res_r3 = cv2.resize(res, (h, w))
            r3 = m + '_r3'
            # cv2.imwrite(os.path.join(yg_block_merge, r3 + '.jpg'), res_r3)


# xml增加换行符
def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def creat_node(fa, name, text=None):
    element = ET.Element(name)
    if not None:
        element.text = text
    fa.append(element)

    return element


def anno_voc(save_path,folder,filename,size,name,bbox):

    root = ET.Element('annotation')
    tree = ET.ElementTree(root)

    creat_node(root, 'folder', folder)
    creat_node(root, 'filename', filename)

    source_root = creat_node(root, 'source')
    creat_node(source_root, 'database', 'Unknow')

    s_str = [str(i) for i in size]
    size_root = creat_node(root, 'size')
    creat_node(size_root, 'width', s_str[0])
    creat_node(size_root, 'height', s_str[1])
    creat_node(size_root, 'depth', s_str[2])

    creat_node(root, 'segmented', '0')

    object_root = creat_node(root, 'object')
    creat_node(object_root, 'name', name)
    creat_node(object_root, 'pose', 'Unspecified')
    creat_node(object_root, 'truncated', '0')
    creat_node(object_root, 'difficult', '0')
    bndbox_root = creat_node(object_root, 'bndbox')

    bbox_str = [str(i) for i in bbox]
    creat_node(bndbox_root, 'xmin', bbox_str[0])
    creat_node(bndbox_root, 'ymin', bbox_str[1])
    creat_node(bndbox_root, 'xmax', bbox_str[2])
    creat_node(bndbox_root, 'ymax', bbox_str[3])

    __indent(root)

    tree.write(save_path, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':

    img_dir = r'E:\DataSet\yingguang_new\yg_merge_src'  # 背景图

    dst_dir = r'E:\DataSet\yingguang_new\yg_merge_dst'  # 生成图
    anno_dir = r'E:\DataSet\yingguang_new\yg_merge_dst_anno'  # 生成xml标注文件

    block_dir = r'E:\DataSet\yingguang_new\yg_block_merge'  # 融合的荧光图
    dirname = 'yg_merge_dst'

    imgnames = os.listdir(img_dir)
    imgpaths = [os.path.join(img_dir, i) for i in imgnames]

    blocks = os.listdir(block_dir)
    blockpaths = [os.path.join(block_dir, i) for i in blocks]
    random.shuffle(blockpaths)

    cnt = len(imgpaths)

    for i in range(cnt):
        print('process: ', imgpaths[i])
        dst = cv2.imread(imgpaths[i])  # 大约(300, 300)
        src = cv2.imread(blockpaths[i])
        h, w, d = dst.shape
        h1, w1, _ = src.shape

         # src_mask = 255 * np.ones(src.shape, src.dtype)
        src_mask = np.zeros(src.shape, src.dtype)

        poly = np.array([[2, 10], [2, h1-10], [5, h1-5], [10,h1-2], [w1-10, h1-2], [w1 - 5, h1 - 5], [w1 - 2, h1 - 10],\
                         [w1-2, 10], [w1 - 5,5], [w1-10, 2], [10, 2]], np.int32)
        cv2.fillPoly(src_mask, [poly], (255, 255, 255))

        #  随机random.randrange(start, stop[, step])
        # w(50,1000) h(50, 300)



        xmin = random.randint(50, w - w1 - 50)
        ymin = random.randint(50, h - h1 - 50)

        center = (xmin + w1 // 2 - 10 , ymin + h1 // 2 - 10)

        # Clone seamlessly.
        output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

        # 图像叠加
        # roi = dst[tl_y: tl_y + h1, tl_x : tl_x + w1]
        # roi = cv2.addWeighted(src, 0.9, roi, 0.1, 0)  # 图像融合
        # add_img = dst.copy()
        # add_img[tl_y: tl_y + h1, tl_x: tl_x + w1] = roi

        # cv2.imshow('seamless', output)
        # cv2.imshow('merge', add_img)
        # cv2.waitKey(0)

        save_img = os.path.join(dst_dir, imgnames[i])
        save_xml = os.path.join(anno_dir, imgnames[i].split('.')[0] + '.xml')

        cv2.imwrite(save_img, output)
        anno_voc(save_xml, dirname, save_img, [w,h,d], '5', [xmin, ymin, xmin + w1, ymin + h1])




