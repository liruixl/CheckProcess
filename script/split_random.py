
import os
import shutil
import random


if __name__ == '__main__':
    src_dir = r'E:\DataSet\hand_check_test\lr\UpG'
    dst_dir = r'E:\DataSet\hand_check_test\lr\UpG_50'

    filenames = os.listdir(src_dir)
    random.shuffle(filenames)
    filenames = filenames[:50]

    for name in filenames:
        shutil.copyfile(os.path.join(src_dir, name), os.path.join(dst_dir, name))
