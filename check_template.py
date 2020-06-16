import cv2
import os
import numpy as np


class CheckTemplate:

    def __init__(self, template_dir):
        self._CT = ['Normal', 'Transfer', 'Cash']


        self.normal = [1330, 630]
        self.template_dir = template_dir
        self.template_img_names =  {0:'date.jpg ', 1:'date_year.jpg ' , 2:'date_month.jpg', 3:'date_day.jpg',
                       4:'bank_fukuan.jpg', 5:'person_shoukuan.jpg', 6:'account_chupiao.jpg ',
                       7:'usage.jpg', 8:'prompt_up.jpg', 9:'prompt_down.jpg', 10:'signature.jpg',
                       11:'pwd.jpg', 12:'bank_number.jpg', 13:'check.jpg ', 14:'record.jpg',
                       15: 'money_chinese.jpg', 16:'money_number.jpg',
                       17:'deadline.jpg'}
        # (1330, 630) [xmin,ymin,xmax,ymax]
        left_up = [60, 100, 400, 320]  # 左上区域
        mid = [350, 100, 760, 180]  # 中间年月日
        right_up = [700, 80, 1300, 300]  # 右上区域

        left_bottom = [30, 250, 400, 530]  # 左下区域
        right_bottom = [700, 240, 1100, 540]  # 右下区域

        left = [0, 100, 120, 530]  # 左侧

        self.temp_bbox = {0: left_up, 5:left_up, 15:left_up,
                          1:mid, 2:mid, 3:mid,
                          4:right_up, 6:right_up, 16:right_up,
                          7:left_bottom, 8:left_bottom, 9:left_bottom, 10:left_bottom,
                          11:right_bottom, 12:right_bottom, 13:right_bottom,14:right_bottom,
                          17:left
                          }

        assert len(self.template_img_names) == len(self.temp_bbox)

        self.template_cvimg = {}
        for id, imgname in self.template_img_names.items():
            img_path = os.path.join(self.template_dir, imgname)
            self.template_cvimg[id] = cv_imread(img_path)


    def get_templates(self, check_type='Normal'):

        if check_type not in self._CT:
            raise NotImplementedError('Can not find %s type check template.' % (check_type))

        if check_type == self._CT[2]:
            temp = self.template_cvimg.copy()
            temp.pop(12)  # 12:'bank_number.jpg'
            return temp
        return self.template_cvimg.copy()


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img