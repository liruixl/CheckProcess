# modified from https://blog.csdn.net/yuanlulu/article/details/82222119
# from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 200
GOOD_MATCH_PERCENT = 0.1

def delBlackPoint(binary):
    tempkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dila = cv2.dilate(binary, tempkernel, iterations=1)
    ero = cv2.erode(dila, tempkernel, iterations=1)
    return ero

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # _, im1Gray = cv2.threshold(im1Gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, im2Gray = cv2.threshold(im2Gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 去除小黑点
    # im1Gray = delBlackPoint(im1Gray)
    # im2Gray = delBlackPoint(im2Gray)


    cv2.imshow('dect gary', im1Gray)
    cv2.imshow('ori gary', im2Gray)
    cv2.waitKey(0)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    print('match1to2', print(type(matches)))
    print(matches)

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None, flags=2)

    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    # queryIdx : 查询点的索引（当前要寻找匹配结果的点在它所在图片上的索引）.
    # trainIdx : 被查询到的点的索引（存储库中的点的在存储库上的索引）
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography https://blog.csdn.net/liuphahaha/article/details/50719275
    # findHomography 函数使用一种被称为随机抽样一致算法(Random Sample Consensus )
    # 的技术在大量匹配错误的情况下计算单应性矩阵。
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # 计算得到转换矩阵
    # h = cv2.getPerspectiveTransform(points1, points2)

    if h is None:
        return im1, None

    # Use homography
    height, width, channels = im2.shape

    # cv2.warpPerspective() 透视变换
    im1Reg = cv2.warpPerspective(im1, h, (width + im1.shape[1], height))

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    refFilename = r'img_hanghui/zhongnong_uv.jpg'
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = r'img_dect/zhongnong_1.jpg'
    print("Reading image to align : ", imFilename);
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(im, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename);
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
