
import cv2
import os
import numpy as np

def fiber_to_binary_samesize():

    fiber_img_dir = r'E:\DataSet\fiber\cluster_exp'
    dst_dir = r'E:\DataSet\fiber\cluster_roi'

    imgname_list = os.listdir(fiber_img_dir)


    for imgname in imgname_list:

        print('process: ', imgname)
        imFilename = os.path.join(os.path.join(fiber_img_dir, imgname))

        im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # _, gray = cv2.threshold(gray, thre, 255, cv2.THRESH_BINARY)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 只寻找最外轮廓 RETR_EXTERNAL RETR_CCOMP
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours.sort(key=lambda cnt : cv2.contourArea(cnt), reverse=True)

        cnt = contours[0]
        im_h, im_w = gray.shape

        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = max(x - 4, 0), max(y - 4, 0), min(w + 8, im_w), min(h + 8, im_h)

        fiber = im[y:y+h, x:x+w]
        fiber_name = os.path.join(dst_dir,imgname)
        cv2.imwrite(fiber_name, fiber)

        # H, W = gray.shape
        # roi = np.zeros((H, W), dtype=np.uint8)
        #
        # # 绘制轮廓
        # cv2.drawContours(roi, contours, -1, 255, -1, lineType=cv2.LINE_AA)
        #
        # cv2.imshow('roi', roi)
        # cv2.waitKey(0)


if __name__ == '__main__':
    fiber_to_binary_samesize()