r"""
小脚本
"""


import os
import shutil

def split_guonan_by_class(byname):
    """
    分离各类变造鉴真点

    复制整个文件夹:
    shutil.copyfile()
    shutil.copytree()
    :return:None
    """
    # src_dir = r'E:\异常图0927\敦南\敦南异常'
    # dst_dir = r'E:\异常图0927\郭南异常细分\团花异常'

    src_dir = r'E:\异常图0927\华菱\异常票'
    dst_dir = r'E:\异常图0927\华菱异常细分\团花异常'

    namelist = os.listdir(src_dir)
    namelist = [i for i in namelist if byname in i]

    for name in namelist:
        shutil.copytree(os.path.join(src_dir, name), os.path.join(dst_dir, name))
        print('copy', name)


if __name__ == '__main__':
    split_guonan_by_class('团花')
    pass