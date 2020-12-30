"""
支票相关类
"""

import os
import cv2
from tools.imutils import cv_imread

class CheckDir:
    def __init__(self):

        # 常用
        self.upg = None
        self.upir = None
        self.upirtr = None
        self.upuv = None

        # 不常用
        self.upuvtr = None
        self.dwg = None
        self.dwir = None
        self.dwirtr = None
        self.dwuv = None
        self.dwuvtr = None

    def init_by_checkdir(self, checkdir_path, ext='.bmp'):
        self.upg    = cv_imread(os.path.join(checkdir_path, 'UpG' + ext))
        self.upir   = cv_imread(os.path.join(checkdir_path, 'Upir' + ext))
        self.upirtr = cv_imread(os.path.join(checkdir_path, 'Upirtr' + ext))
        self.upuv   = cv_imread(os.path.join(checkdir_path, 'Upuv' + ext))
        self.upuvtr = cv_imread(os.path.join(checkdir_path, 'Upuvtr' + ext))
        self.dwg    = cv_imread(os.path.join(checkdir_path, 'DwG' + ext))
        self.dwir   = cv_imread(os.path.join(checkdir_path, 'Dwir' + ext))
        self.dwirtr = cv_imread(os.path.join(checkdir_path, 'Dwirtr' + ext))
        self.dwuv   = cv_imread(os.path.join(checkdir_path, 'Dwuv' + ext))
        self.dwuvtr = cv_imread(os.path.join(checkdir_path, 'Dwuvtr' + ext))

    def set_upg(self, img):
        self.upg = img

    def set_upir(self, img):
        self.upir = img

    def set_upirtr(self, img):
        self.upirtr = img

    def set_upuv(self, img):
        self.upuv = img



if __name__ == '__main__':

    checkpath = r'E:\异常图0927\敦南\敦南正常\20200828_15-40-36'

    check_dir = CheckDir()
    check_dir.init_by_checkdir(checkpath)

    cv2.imshow('a', check_dir.upuv)
    cv2.imshow('b', check_dir.upir)
    cv2.imshow('c', check_dir.upirtr)
    cv2.imshow('d', check_dir.upg)

    cv2.waitKey(0)