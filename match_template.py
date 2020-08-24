import os
import cv2 as cv
import numpy as np
import json
import time
import re

from check_template import CheckTemplate

from findisExist import isExist, custom_threshold_old, constract_brightness, threshold_demo
from red_waterline.extract_red2 import way_3

from tools.imutils import cv_imread
# 检测红外保护油墨字体
# 20200616 包含红水线

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)

color_list = color_list.reshape((-1, 3)) * 255

colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 3)

threshold = 0.8

check_scale = (1330, 630)
template_dir = r'E:\YHProject\票据处理\ch\parts_1392_628'


# 全局变量，也可以传入模板匹配函数中
check_type = ['Normal', 'Transfer', 'Cash']
check_type = 'Cash'
check_temps = CheckTemplate(template_dir)
template_images = check_temps.get_templates(check_type)
areas = check_temps.temp_bbox


work_dir = r"C:\Users\lirui\Desktop\temp\check"
target_img = os.path.join(work_dir, '092339_Upir.jpg')
target_img = r'C:\Users\lirui\Desktop\temp\check\Upir_nonghang_zhuangzhang.jpg'

def template_match(upg_img_path, upir_img_path, method=cv.TM_CCOEFF_NORMED):

    # target = cv.imread(upg_img_path)
    # target_upir = cv.imread(upir_img_path)

    target = cv_imread(upg_img_path)
    target_upir = cv_imread(upir_img_path)

    binary_target = threshold_demo(target)
    target_upir = constract_brightness(target_upir, 1.8, 1)
    target_upir = custom_threshold_old(target_upir)

    cropx_offset = 0

    ori_h, ori_w = target.shape[:2]
    if ori_w > check_scale[0]:
        cropx_offset = ori_w - check_scale[0]
        target = target[:, cropx_offset:]  # 裁剪坐标为[y0:y1, x0:x1]
        binary_target = binary_target[:, cropx_offset:]
        target_upir = target_upir[:, cropx_offset:]

    # print(target.shape)

    ids = []
    locs = []
    scores = []
    isinks = []

    time_start = time.time()

    for key, cvimg in template_images.items():
        # 1 取模板
        temp = cvimg
        t_height, t_width = temp.shape[:2]  # row h, col w

        # 2 划可能的区域
        possible_area = areas[key]
        xmin, ymin, xmax, ymax = possible_area
        target_crop = target[ymin:ymax, xmin:xmax]

        result = cv.matchTemplate(target_crop, temp, method)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            t_left = min_loc
        else:
            t_left = max_loc  # (x,y) 访问 result[y][x], row col

        # 3 将坐标映射回裁掉小票后的标准支票原图
        tl = (t_left[0]+xmin, t_left[1]+ymin)
        br = (tl[0]+t_width, tl[1]+t_height)

        score = '{0:.3f}'.format(result[t_left[1]][t_left[0]])
        confidence = isExist(binary_target,target_upir,tl,br)

        # cv.rectangle(target, tl, br, colors[key].tolist(), -1)

        ids.append(key)
        # 偏移量
        locs.append([tl[0]+cropx_offset, tl[1], br[0]+cropx_offset, br[1]])
        scores.append(float(score))
        isinks.append(confidence)


    # 红水线位置可根据 人名币大写 字段来判断位置
    # 15: 'money_chinese.jpg'
    idx = ids.index(15)
    tl = (locs[idx][0], locs[idx][1])
    br = (locs[idx][2], locs[idx][3])

    top_right = (br[0], tl[1])

    xmin, ymin = redline_tl = (top_right[0] + 4, top_right[1] - 6)
    xmax, ymax = redline_br = (redline_tl[0] + 666, redline_tl[1] + 66)

    redline_img = target[ymin: ymax, xmin:xmax]
    damage = way_3(redline_img, debug=False) # 展示红水线晕染、刮擦白墨团等检测结果
    ids.append(len(ids))
    locs.append([xmin, ymin, xmax, ymax])
    scores.append(1.0)
    isinks.append(1 - damage)

    time_end = time.time()

    # print('cost time:', time_end-time_start)
    return ids, locs, scores, isinks



def template_demo():
    work_dir = r'E:\YHProject\CheckProcess\img'
    temp_img = os.path.join(work_dir, 'cut1.png')
    temp_img = os.path.join(work_dir, r'huanhui_temp.jpg')

    target_img = r'E:\YHProject\票据处理\各行支票汇总\07.东莞银行\清分支票\02\Upuv.jpg'

    temp = cv.imread(temp_img)
    # temp = temp[:, :, :3]
    target = cv_imread(target_img)
    print(temp.shape)
    print(target.shape)
    cv.imshow("template", temp)
    cv.imshow("target", target)

    cv.waitKey(0)
    # cv.imshow("target", target)
    # 3种模板匹配方法
    # methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]

    # 6 种
    # methods = ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED',
    #            'cv.TM_CCORR','cv.TM_CCORR_NORMED',
    #            'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED']

    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    methods = [cv.TM_CCORR_NORMED]

    t_height, t_width = temp.shape[:2]

    for method in methods:
        # method = eval(meth)
        print(method)
        result = cv.matchTemplate(target, temp, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        # 获得目标图中左上角匹配位置
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            t_left = min_loc
            print(min_val, end=':')
        else:
            t_left = max_loc
            print(max_val, end=':')
        # 获得目标图中右下角位置
        print(t_left)
        br = (t_left[0]+t_width, t_left[1]+t_height)

        target_copy = target.copy()

        cv.rectangle(target_copy, t_left, br, (0, 0, 255), 2)  # red

        score = '{0:.3f}'.format(result[t_left[1]][t_left[0]])
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(target_copy, score, (t_left[0], t_left[1] - 2),
                   font, 0.5, (0, 255, 0), thickness=1, lineType=cv.LINE_AA)  # green

        cv.imshow("matching-"+np.str(method), target_copy)
        cv.waitKey(0)


def template_multi_target():
    temp_img = os.path.join(work_dir, 'cut1.png')
    temp = cv.imread(temp_img)
    target = cv.imread(target_img)
    cv.imshow("template", temp)

    # 3 种归一化的
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]

    t_height, t_width = temp.shape[:2]

    for method in [5]:
        print(method)
        result = cv.matchTemplate(target, temp, method)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        # 获得目标图中左上角匹配位置
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            t_left = min_loc
            print(min_val, end=':')
            loc = np.where(result <= 1 - threshold)

        else:
            t_left = max_loc
            print(max_val, end=':')
            loc = np.where(result >= threshold)

        print(loc)
        target_copy = target.copy()

        for pt in zip(*loc[::-1]):  # *号表示可选参数
            print(pt)
            br = (pt[0] + t_width, pt[1] + t_height)
            cv.rectangle(target_copy, pt, br, (0, 0, 255), 2) # rgb
            # 图片，添加的文字，左上角坐标(整数)，字体，字体大小，颜色，字体粗细
            cv.putText(target_copy, str(result[pt[1]][pt[0]]), (pt[0], pt[1] - 7), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 1)

        cv.imshow("matching-" + np.str(method), target_copy)


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv_img=cv.cvtColor(cv_img,cv.COLOR_RGB2BGR)
    return cv_img


def demo(img_path, img_ir_path, debug=False, save_dir=None):
    ids, locs, scores, isinks = template_match(img_path, img_ir_path)

    # print(ids)
    # print(locs)
    # print(scores)
    # print(isinks)

    image_name = re.split('\\\\+|/+', img_path)[-1].split('.')[0]
    print('detect ink: ', image_name)
    if save_dir is not None:
        save_json = {}
        for i in range(len(ids)):
            save_json[ids[i]] = {'loc': locs[i], 'score': scores[i], 'confidence': isinks[i]}
        with open(os.path.join(save_dir, image_name + '.json'), 'w') as f:
            json.dump(save_json, f, indent=4)

    if debug:
        target = cv_imread(img_path)
        target_ir = cv_imread(img_ir_path)


        for idx in range(len(ids)):
            key = ids[idx]
            score = scores[idx]
            is_ink = isinks[idx]

            xmin, ymin, xmax, ymax = locs[idx]
            tl = (xmin, ymin)
            br = (xmax, ymax)

            # if idx ==16:
            #     dst = target[ymin:ymax, xmin:xmax]
            #     dst_ir = target_ir[ymin:ymax, xmin:xmax]
            #     cv.imshow("crop", dst)
            #     cv.imshow('crop_ir', dst_ir)
            #     cv.waitKey(0)

            cv.rectangle(target, tl, br, colors[key].tolist(), 1)
            font = cv.FONT_HERSHEY_SIMPLEX
            text = '{0}:{1} {2}'.format(key, score, is_ink)
            cv.putText(target, text, (tl[0], tl[1] - 4),
                       font, 0.5, (0, 255, 0), thickness=1, lineType=cv.LINE_AA)

        if save_dir is not None:
            save_name = os.path.join(save_dir, img_path.split('\\')[-1])
            cv.imwrite(save_name, target)

        cv.imshow("match_template", target)
        cv.waitKey(0)
        cv.destroyAllWindows()





def tst_check100():
    zong = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\check_UpG_Upir'
    # img_path = os.path.join(zong, '092723_UpG.jpg')
    # img_ir_path = os.path.join(zong, '092723_Upir.jpg')

    save_dir = r'C:\Users\lirui\Desktop\temp\check100_match_res'

    imgs = os.listdir(zong)

    for img in imgs:
        if 'UpG' in img:
            print('process:' + img)
            img_path = os.path.join(zong, img)
            img_ir_path = os.path.join(zong, img.split('_')[0] + '_Upir.jpg')

            demo(img_path, img_ir_path, debug=True, save_dir=save_dir)


if __name__ == '__main__':
    # src = cv.imread(target_img, cv.IMREAD_COLOR)
    # cv.namedWindow("check", cv.WINDOW_AUTOSIZE)
    # cv.imshow("check", src)

    # img_path = r'C:\Users\lirui\Desktop\temp\check\UpG_nonghang_zhuanzhang.jpg'
    # img_ir_path = r'C:\Users\lirui\Desktop\temp\check\Upir_nonghang_zhuangzhang.jpg'

    # zong = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\check_UpG_Upir'
    # img_path = os.path.join(zong, '092342_UpG.jpg')
    # img_ir_path = os.path.join(zong, '092342_Upir.jpg')

    # =====
    # img_path = r'E:\DataSet\hand_check_test\ch\UpG\UpG_153614.bmp'
    # img_ir_path = r'img\Upir_1.jpg'
    #
    # save_dir =  r'result_json'
    #
    # demo(img_path,img_ir_path,debug=True,save_dir=save_dir)


    template_demo()