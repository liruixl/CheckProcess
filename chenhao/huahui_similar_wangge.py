import cv2
import similar_test as similar_util

#竹子：
zhuzi = [
    [[450, -135], [317, -258], [90]],
    [[94, -225], [-7, -282], [80]],
    [[-295, -124], [-391, -197], [85]],
    [[345, 68], [204, -4], [80]],
    [[196, 55], [13, -8], [80]],
    [[278, -242], [107, -320], [85]],
    [[-224, 2], [-359, -105], [80]]
]

meihua = [
    [[2, -137], [-100, -201], [85]],
    [[-72, -24], [-153, -62], [75]],
    [[-795, -74], [-872, -147], [85]],
    [[-230, -47], [-320, -106], [85]],
    [[1, -259], [-161, -331], [80]],
    [[-553, -226], [-615, -274], [75]],
    [[-730, -301], [-819, -335], [80]],
    [[-424, -136], [-528, -180], [75]]
]


#zhuzhi
#横坐标：915 ~ 1764 [849]  纵坐标：277 ~ 612 [335]
#截成55x55的矩形区域
#横15格，纵6格

#meihua
#横坐标：880 ~ 1762 [882]  纵坐标：278 ~ 605 [327]
#截成55x55的矩形区域
#横15格，纵6格

zhuzi_template = cv2.imread("img/template/zhuzi_3.png")
meihua_template = cv2.imread("img/template/meihua.png")

def getZhuziStandardCoor(target_image):
    theight, twidth = zhuzi_template.shape[:2]
    # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(target_image, zhuzi_template, cv2.TM_SQDIFF_NORMED)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return min_loc

def getMeihuaStandardCoor(target_image):
    theight, twidth = meihua_template.shape[:2]
    # 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(target_image, meihua_template, cv2.TM_SQDIFF_NORMED)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # strmin_val = str(min_val)
    # # 绘制矩形边框，将匹配区域标注出来
    # # min_loc：矩形定点
    # # (min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
    # # (0,0,225)：矩形的边框颜色；2：矩形边框宽度
    # cv2.rectangle(target_image, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)
    # # 显示结果,并将匹配值显示在标题栏上
    # cv2.imshow("MatchResult----MatchingValue=" + strmin_val, target_image)
    # cv2.waitKey()
    return min_loc


zhuzi_norImgList = []
meihua_norImgList = []

zhuzi_row_count = 6  #横向分为多少格
zhuzi_col_count = 15   #纵向分为多少格
zhuzi_row_pixel = 335 // zhuzi_row_count
zhuzi_col_pixel = 849 // zhuzi_col_count

meihua_row_count = 6  #横向分为多少格
meihua_col_count = 15   #纵向分为多少格
meihua_row_pixel = 327 // meihua_row_count
meihua_col_pixel = 882 // meihua_col_count


def get_zhuziNormImg(normal_img):
    # norImgList = []
    min_loc = getZhuziStandardCoor(normal_img)
    for i in range(zhuzi_row_count):
        y1 = min_loc[1] + zhuzi_row_pixel * i
        y2 = min_loc[1] + zhuzi_row_pixel * (i + 1)
        for j in range(zhuzi_col_count):
            x1 = min_loc[0] + zhuzi_col_pixel * j
            x2 = min_loc[0] + zhuzi_col_pixel * (j + 1)
            block_img = normal_img[y1:y2, x1:x2]
            zhuzi_norImgList.append(block_img)
    return zhuzi_norImgList

def get_meihuaNormImg(normal_img):
    # norImgList = []
    min_loc = getMeihuaStandardCoor(normal_img)
    for i in range(meihua_row_count):
        y1 = min_loc[1] + meihua_row_pixel * i
        y2 = min_loc[1] + meihua_row_pixel * (i + 1)
        for j in range(meihua_col_count):
            x1 = min_loc[0] + meihua_col_pixel * j
            x2 = min_loc[0] + meihua_col_pixel * (j + 1)
            block_img = normal_img[y1:y2, x1:x2]
            meihua_norImgList.append(block_img)
    return meihua_norImgList


#竹子
def getZhuzi(image_test):
    correct_num = 0
    total_num = zhuzi_row_count * zhuzi_col_count
    err_corri = []
    similar_list = []
    min_loc = getZhuziStandardCoor(image_test)
    #过滤条件改改
    if image_test.shape[1] - min_loc[0] >= 1049:
        return 0.0, [], []

    block_index = 0 #第几个格子
    for i in range(zhuzi_row_count):
        y1 = min_loc[1] + zhuzi_row_pixel * i
        y2 = min_loc[1] + zhuzi_row_pixel * (i + 1)
        for j in range(zhuzi_col_count):
            print("block_index: ", block_index)
            x1 = min_loc[0] + zhuzi_col_pixel * j
            x2 = min_loc[0] + zhuzi_col_pixel * (j + 1)
            block_imgTest = image_test[y1:y2, x1:x2]



            similar = similar_util.similar_calc(block_imgTest, zhuzi_norImgList[block_index])
            similar_list.append(similar)

            cv2.namedWindow("block_imgTest", cv2.WINDOW_NORMAL)
            cv2.imshow('block_imgTest', block_imgTest)

            cv2.namedWindow("block_imgNorm", cv2.WINDOW_NORMAL)
            cv2.imshow('block_imgNorm', zhuzi_norImgList[block_index])

            cv2.waitKey()

            if similar * 100 > 70:
                correct_num += 1
            else:
                err_corri.append([[x1, y1],[x2, y2],[block_index]])
            block_index += 1
    return correct_num / total_num, similar_list, err_corri

def getMeihua(image_test):
    correct_num = 0
    total_num = meihua_row_count * meihua_col_count
    err_corri = []
    similar_list = []
    min_loc = getMeihuaStandardCoor(image_test)
    if image_test.shape[1] - min_loc[0] >= 1018:
        return 0.0, [], []
    block_index = 0  # 第几个格子
    for i in range(meihua_row_count):
        y1 = min_loc[1] + meihua_row_pixel * i
        y2 = min_loc[1] + meihua_row_pixel * (i + 1)
        for j in range(meihua_col_count):
            print("block_index: ", block_index)
            x1 = min_loc[0] + meihua_col_pixel * j
            x2 = min_loc[0] + meihua_col_pixel * (j + 1)
            block_imgTest = image_test[y1:y2, x1:x2]

            similar = similar_util.similar_calc(block_imgTest, meihua_norImgList[block_index])
            similar_list.append(similar)

            cv2.namedWindow("block_imgTest", cv2.WINDOW_NORMAL)
            cv2.imshow('block_imgTest', block_imgTest)

            cv2.namedWindow("block_imgNorm", cv2.WINDOW_NORMAL)
            cv2.imshow('block_imgNorm', meihua_norImgList[block_index])

            cv2.waitKey()

            if similar * 100 > 50:
                correct_num += 1
            else:
                err_corri.append([[x1, y1], [x2, y2], [block_index, similar]])
            block_index += 1
    return correct_num / total_num, similar_list, err_corri


def huahui_confidence(image, flag):
    if flag == 1:
        return getZhuzi(image)
    elif flag == 2:
        return getMeihua(image)

if __name__ == '__main__':
    img1 = cv2.imread('img/huahui_test/meihua_10.bmp')
    zhuzi_normal = cv2.imread("img/standard_img/zhuzi.jpg")
    meihua_normal = cv2.imread("img/standard_img/meihua.jpg")
    get_zhuziNormImg(zhuzi_normal)
    get_meihuaNormImg(meihua_normal)
    confidence, similar_list, err_coori = huahui_confidence(img1, 2)
    print(similar_list)
    print("花卉整体置信度：", confidence)
    if len(err_coori) > 0:
        print("疑似造假区域：" )
        for i in range(len(err_coori)):
            coori = err_coori[i]
            y1 = coori[0][1]
            y2 = coori[1][1]
            x1 = coori[0][0]
            x2 = coori[1][0]
            err_img = img1[y1:y2, x1:x2]
            print("index:", coori[2][0], ", 坐标：", coori)
            cv2.namedWindow("err_img", cv2.WINDOW_NORMAL)
            cv2.imshow('err_img', err_img)
            cv2.waitKey()