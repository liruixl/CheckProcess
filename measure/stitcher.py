import cv2
import numpy as np
from red_waterline.union_find import UnionFind

class Stitcher:

    def __init__(self):
        self.methods = ['orb', 'sift', 'surf']
        self.debug = False

    def stitch(self, dectect_img, normal_img, ratio=0.75, reproThresh=4, showMathes=False):

        # imgaeB : 模板图,即正常图像
        (imageB, imageA) = normal_img, dectect_img
        # 第一步：计算kpsA和dpsA
        method = self.methods[2]
        (kpsA, dpsA) = self.detectandcompute(imageA, method=method)
        (kpsB, dpsB) = self.detectandcompute(imageB, method=method)

        # 获得变化的矩阵H
        M = self.matchKeypoint(kpsA, dpsA, kpsB, dpsB, ratio, reproThresh)

        if M is None:
            print('Oh!! No!! match failure')
            return None
        (matches, H, status) = M

        print('match point:', len(kpsA))
        print('match point:', len(kpsB))

        print('matches:', len(matches))

        off_x, off_y = self.find_offset(kpsA, kpsB, matches, status)

        # 第四步：使用cv2.warpPerspective获得经过H变化后的图像 1w 0h
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageB.shape[0]))
        aligend = result[0:imageB.shape[0], 0:imageB.shape[1]]

        if self.debug:
            cv2.imshow('detect img', dectect_img)
            cv2.imshow('normal img', normal_img)

            cv2.imshow('detect warpPerspective', result)
            cv2.imshow('aligend', aligend)
            cv2.waitKey(0)

        # 第五步：将图像B填充到进过H变化后的图像，获得最终的图像
        # result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        if showMathes:
            # 第六步：对图像的关键点进行连接
            via = self.showMatches(imageA, imageB, kpsA, kpsB, matches, status)

        return off_x, off_y


    def find_offset(self, kpsA, kpsB, matches, status, offset=5):
        ptAs = []
        ptBs = []  # B代表的是模板图像，查询queryImg

        # 一般来说，检测的范围要比模板图像的范围大，否则就是检测不全
        # 提供正确估计的好的匹配被叫做inliers
        for (trainIdx, queryIdx), s in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]), int(kpsB[trainIdx][1]))
                ptAs.append(ptA)
                ptBs.append(ptB)

        n = len(ptAs)
        print('inliers points:', n)

        uf = UnionFind(n)

        for i in range(n):
            for j in range(i + 1, n):
                pp1 = [ptAs[i][0], ptAs[i][1], ptBs[i][0], ptBs[i][1]]  # like[xmin, ymin, xmax, ymax]
                pp2 = [ptAs[j][0], ptAs[j][1], ptBs[j][0], ptBs[j][1]]

                off1 = (pp1[2] - pp1[0], pp1[3] - pp1[1])
                off2 = (pp2[2] - pp2[0], pp2[3] - pp2[1])

                if abs(off2[0] - off1[0]) < offset and abs(off2[1] - off1[1]) < offset:
                    uf.union(i, j)

        point_set = dict()
        for i in range(n):
            if uf.find(i) == i:
                point_set[i] = []

        for i in range(n):
            p = uf.find(i)
            point_set[p].append(i)

        most_point_id = -1
        size = -1

        for id, ps in point_set.items():
            if len(ps) > size:
                size = len(ps)
                most_point_id = id

        print('points are splited to set count:', uf.get_count())
        print('most point set id:', most_point_id, ' point nums:', len(point_set[most_point_id]))

        pps = point_set[most_point_id]
        off_x = 0
        off_y = 0
        for i in range(len(pps)):
            p2p = [ptAs[i][0], ptAs[i][1], ptBs[i][0], ptBs[i][1]]
            off_x += (p2p[0] - p2p[2])
            off_y += (p2p[1] - p2p[3])

        off_x = off_x // len(pps)
        off_y = off_y // len(pps)

        print('相对原图的偏移量为', off_x, off_y)
        return off_x, off_y

    def showMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 将两个图像进行拼接
        # 根据图像的大小，构造全零矩阵
        via = np.zeros((max(imageB.shape[0], imageA.shape[0]), imageA.shape[1] + imageB.shape[1], 3), np.uint8)
        # 将图像A和图像B放到全部都是零的图像中
        via[0:imageA.shape[0], 0:imageA.shape[1]] = imageA
        via[0:imageB.shape[0], imageA.shape[1]:] = imageB
        # 根据matches中的索引，构造出点的位置信息
        for (trainIdx, queryIdx), s in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0] + imageA.shape[1]), int(kpsB[trainIdx][1]))
                # 使用cv2.line进行画图操作
                # cv2.line(via, ptA, ptB, (0, 255, 0), 1)

                cv2.circle(via, ptA, 5, (0, 0, 255), 1)
                cv2.circle(via, ptB, 5, (0, 0, 255), 1)
        cv2.imshow('line draw', via)
        cv2.waitKey(0)

        return via

    def matchKeypoint(self, kpsA, dpsA, kpsB, dpsB, ratio, reproThresh):

        # 第二步：实例化BFM匹配， 找出符合添加的关键点的索引
        bf = cv2.BFMatcher()

        matcher = bf.knnMatch(dpsA, dpsB, 2)

        matches = []

        for match in matcher:

            if len(match) == 2 and match[0].distance < match[1].distance * ratio:
                # 加入match[0]的索引
                matches.append((match[0].trainIdx, match[0].queryIdx))
        # 第三步：使用cv2.findHomography找出符合添加的H矩阵
        if len(matches) > 4:
            # 根据索引找出符合条件的位置
            kpsA = np.float32([kpsA[i] for (_, i) in matches])
            kpsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(kpsA, kpsB, cv2.RANSAC, reproThresh)

            return (matches, H, status)
        return None

    def detectandcompute(self, image, method='sift'):
        # 进行灰度值转化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == 'sift':
            # 实例化sift函数
            sift = cv2.xfeatures2d.SIFT_create()
            # 获得kps关键点和dps特征向量sift
            kps, dps = sift.detectAndCompute(gray, None)
            # 获得特征点的位置信息， 并转换数据类型
            kps = np.float32([kp.pt for kp in kps])
            return (kps, dps)

        elif method == 'surf':
            surf = cv2.xfeatures2d_SURF.create()
            kps, dps = surf.detectAndCompute(gray, None)
            kps = np.float32([kp.pt for kp in kps])
            return (kps, dps)

        elif method == 'orb':
            orb = cv2.ORB_create()
            kps, dps = orb.detectAndCompute(gray, None)
            kps = np.float32([kp.pt for kp in kps])
            return (kps, dps)

        raise RuntimeError('Unknow feature tiqu method')


if __name__ == '__main__':
    # 行徽模板
    # refFilename = r'../hanghui/img_hanghui/zhongnong_uv.jpg'

    # 团花模板
    refFilename = r'../hanghui/img_tuanhua/tuanhua_uv_1.jpg'

    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = r'../hanghui/img_tuanhua/test_1.JPG'

    # imFilename = r'../hanghui/img_dect/tuanhua_2.jpg'

    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)


    stitcher = Stitcher()
    stitcher.debug = True
    stitcher.stitch(im, imReference, showMathes=True)