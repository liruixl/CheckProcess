import os
import shutil

check100_dir = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\100张支票图片'
check_ir_dir = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\100张支票数据之红外水印'

IR_dir = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\check100_split\IR'
UV_dir = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\check100_split\UV'
G_dir = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\check100_split\G'

UP_IR_dir = r'C:\Users\lirui\Desktop\票据处理\图片数据20191114\check_UpG_Upir'

IR_names = ['Dwir','Dwirtr','Upir','Upirtr']
UP_IR_names = ['UpG' ,'Upir']


def split_up_ir(check_dir, des_dir):
    all_dir = os.listdir(check_dir)
    for one_dir in all_dir:
        dir_path = os.path.join(check_dir, one_dir)
        if os.path.isdir(dir_path):  # 保存图像的支票目录，每个目录保存一张支票的10个图像
            all_files = os.listdir(dir_path)

            for file in all_files:
                sp = file.split('.')

                if len(sp) > 1 and (sp[1] == 'jpg') and (sp[0] in UP_IR_names):
                    print('move ' + file + ' to ' + os.path.join(des_dir, sp[0]))
                    # shutil.copy(os.path.join(dir_path, file), os.path.join(des_dir, file.split('.')[0]))
                    new_filename = des_dir + '\\' + one_dir + '_' + sp[0] + '.jpg'
                    shutil.copyfile(os.path.join(dir_path, file), new_filename)
                else:
                    print(file + ' is not UP IR img')

def split_IR(check_dir, des_dir):
    all_dir = os.listdir(check_dir)
    for one_dir in all_dir:
        dir_path = os.path.join(check_dir,one_dir)
        if os.path.isdir(dir_path): # 保存图像的支票目录，每个目录保存一张支票的10个图像
            all_files = os.listdir(dir_path)

            for file in all_files:
                sp = file.split('.')

                if len(sp) > 1 and (sp[1] == 'jpg') and (sp[0] in IR_names):
                    print('move ' + file + ' to ' + os.path.join(des_dir,sp[0]))
                    # shutil.copy(os.path.join(dir_path, file), os.path.join(des_dir, file.split('.')[0]))
                    new_filename = os.path.join(des_dir, sp[0]) + '\\' + sp[0] + '_' + one_dir + '.jpg'
                    shutil.copyfile(os.path.join(dir_path, file),new_filename)
                else:
                    print(file + ' is not IR img')


def split(check_dir, des_dir, prefix):
    # check_dir : 100张支票图片\092339\UpG.jpg
    # des_dir : 复制的目标目录
    # prefix : 分离的前缀名list，例如： ['UpG' ,'Upir']
    # 注意后缀可能变化
    all_dir = os.listdir(check_dir)
    for one_dir in all_dir:
        dir_path = os.path.join(check_dir,one_dir)
        if os.path.isdir(dir_path): # 保存图像的支票目录，每个目录保存一张支票的10个图像
            all_files = os.listdir(dir_path)

            for file in all_files:
                sp = file.split('.')

                if len(sp) > 1 and (sp[1] in ['bmp', 'jpg']) and (sp[0] in prefix):
                    print('move ' + file + ' to ' + os.path.join(des_dir,sp[0]))
                    # shutil.copy(os.path.join(dir_path, file), os.path.join(des_dir, file.split('.')[0]))
                    new_filename = os.path.join(des_dir, sp[0]) + '\\' + sp[0] + '_' + one_dir + '.' + sp[1]
                    shutil.copyfile(os.path.join(dir_path, file), new_filename)
                else:
                    print(file + ' is disgard')



if __name__ == '__main__':
    # split_IR(check_ir_dir,IR_dir)
    # split_up_ir(check100_dir, UP_IR_dir)

    # check_dir = r'E:\DataSet\瑕疵样本数据'

    check_dir = r'E:\DataSet\hand_check_test\100'

    dst_dir = r'E:\DataSet\hand_check_test\lr'
    prefix = ['UpG']

    # check_dir : xxx\092339\UpG.jpg
    # des_dir : 复制的目标目录
    # prefix : 分离的前缀名list，例如： ['UpG' ,'Upir']
    # 注意后缀可能变化****************************************
    # 去splite修改
    split(check_dir,dst_dir,prefix)