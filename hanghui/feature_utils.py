import cv2
import numpy as np

# 输入灰度图

def sift(img):
    sift = cv2.xfeatures2d_SIFT.create()
    keyPoint, descriptor = sift.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
    return keyPoint, descriptor


def surf(img):
    surf = cv2.xfeatures2d_SURF.create()
    keyPoint, descriptor = surf.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
    return keyPoint, descriptor


def orb(img):
    orb = cv2.ORB_create()
    keyPoint, descriptor = orb.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
    return keyPoint, descriptor


def getblackVal(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set threshold level
    threshold_level = 100

    # Find coordinates of all pixels below threshold
    coords = np.column_stack(np.where(gray <= threshold_level))

    # print(coords)

    # Create mask of all pixels lower than threshold level
    mask = gray <= threshold_level

    # Color the pixels in the mask
    image[mask] = (255, 255, 255)

    cv2.imshow('test', image)
    cv2.waitKey(0)


    return mask



def compare(img):
    imgs = []
    keyPoint = []
    descriptor = []

    keyPoint_temp, descriptor_temp = sift(img)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)

    keyPoint_temp, descriptor_temp = surf(img)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)

    keyPoint_temp, descriptor_temp = orb(img)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)
    return imgs, keyPoint, descriptor


def main1():
    method = ['sift','surf','orb']
    img = cv2.imread(r'img_hanghui/zhongnong_uv.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgs, kp, des = compare(gray)

    draw_imgs = []
    for i in range(3):
        img = cv2.drawKeypoints(imgs[i], kp[i], None)
        draw_imgs.append(img)

    cv2.imshow('duibitu', np.hstack(draw_imgs))

    cv2.waitKey()
    cv2.destroyAllWindows()
    print("sift len of des: %d, size of des: %d" % (len(des[0]), len(des[0][0])))
    print("surf len of des: %d, size of des: %d" % (len(des[1]), len(des[1][0])))
    print("orb len of des: %d, size of des: %d" % (len(des[2]), len(des[2][0])))


def main2():
    ppp = r'img_tuanhua/tuanhua_rgb_1.jpg'
    img = cv2.imread(ppp)
    getblackVal(img)

if __name__ == '__main__':

    main2()