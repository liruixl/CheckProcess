
import os
import cv2

if __name__ == '__main__':
    src_dir = r'E:\DataSet\hand_check_test\lr\Dwuv'
    dst_dir = r'E:\DataSet\hand_check_test\Upuv2_test\images'

    for name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, name)
        img = cv2.imread(img_path)

        dst_path = os.path.join(dst_dir, name.split('.')[0] + '.jpg')

        cv2.imwrite(dst_path, img)