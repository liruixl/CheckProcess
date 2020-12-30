import os
import json
import collections

def init_func():
    '''
    初始化dict
    :return: 每个鉴真点的dict
    '''
    jianzhen_dict = collections.OrderedDict()
    name_list = ['kcb', 'bhym', 'hsx', 'rmb', 'zp', 'yghh', 'ygth', 'hlxws', 'ygzthh', 'ygcz']
    for i in range(10):

        res = {
            'loushi_count': 0,              # 漏识数量
            'abnormal_samples_count': 0,    # 异常样本数量
            'wushi_count': 0,               # 误识数量
            'normal_samples_count': 0,      # 正常样本数量
            'wushu_rate': 0.0,              # 误识率
            'loushi_rate': 0.0              # 漏识率
        }
        jianzhen_dict[name_list[i]] = res
    return jianzhen_dict

def acc_statistic(test_res_dir, ground_truth_dir):
    '''
    统计最终的漏识率与误识率
    :param test_res_dir: 测试结果文件夹
    :param ground_truth_dir: 正确标注文件夹
    '''

    res_dict = init_func()

    # 遍历测试结果文件夹
    for root, dirs, files in os.walk(test_res_dir):
        # print(root)
        # print(dirs)
        try:
            for f_name in files:
                test_file = open(os.path.join(root, f_name), 'r')
                ground_truth_file  = open(os.path.join(ground_truth_dir, f_name), 'r')
                test_dict= json.load(test_file)                     #测试结果dict
                ground_truth_dict = json.load(ground_truth_file)    #ground_truth dict
                # 遍历每一个鉴真点的结果，进行统计
                for key in test_dict:
                    if key == 'name':
                        continue
                    if ground_truth_dict[key] == 0:                   # 如果鉴真点样本为假
                        res_dict[key]['abnormal_samples_count'] += 1  # 异常样本数量加1
                        if test_dict[key] == 1:
                            res_dict[key]['loushi_count'] += 1        # 如果存在漏识，数量加1

                    elif ground_truth_dict[key] == 1:                 # 如果鉴真点样本为真
                        res_dict[key]['normal_samples_count'] += 1    # 正常样本数量加1
                        if test_dict[key] == 0:
                            res_dict[key]['wushi_count'] += 1         # 如果存在误识，数量加1
                test_file.close()
                ground_truth_file.close()
        except:
            print('error')


    #计算每个鉴真点的漏识和误识率
    for key in res_dict:
        if res_dict[key]['abnormal_samples_count'] != 0:
            res_dict[key]['loushi_rate'] = res_dict[key]['loushi_count'] / res_dict[key]['abnormal_samples_count']
        if res_dict[key]['normal_samples_count'] != 0:
            res_dict[key]['wushu_rate'] = res_dict[key]['wushi_count'] / res_dict[key]['normal_samples_count']

    print(res_dict)
    f = open('test.json', 'w')
    json.dump(res_dict, f, indent=4, sort_keys=False)
    f.close()


def acc_statistic2(test_res_dir, ground_truth_dir):
    '''

    :param test_res_dir: 测试结果文件夹
    :param ground_truth_dir: 正确标注文件夹
    :return:
    '''

    # kcb_res = {}
    # bhym_res = {}
    # bhym_res = {}
    # hsx_res = {}
    # rmb_res = {}
    # zp_res = {}
    # yghh_res = {}
    # ygth_res = {}
    # hlxws_res = {}
    # ygzthh_res = {}
    # ygcz_res = {}

    # loushi_count = 0                # 漏识数量
    # abnormal_samples_count = 0      # 异常样本数量
    # wushi_count = 0                 # 误识数量
    # normal_samples_count = 0      # 正常样本数量

    res_dict = init_func()

    # 遍历测试结果文件夹
    for root, dirs, files in os.walk(test_res_dir):
        # print(root)
        # print(dirs)
        try:
            for f_name in files:
                test_file = open(os.path.join(root, f_name), 'r')
                ground_truth_file  = open(os.path.join(ground_truth_dir, f_name), 'r')
                test_lines = test_file.readlines()                  #测试结果list
                ground_truth_lines = ground_truth_file.readlines()  #ground_truth list
                # 遍历每一个鉴真点的结果，进行统计
                for i in range(1, len(test_lines)):
                    gt =  ground_truth_lines[i].split(':')[1].strip(' ')
                    ts =  test_lines[i].split(':')[1].strip(' ')
                    if gt == '0':
                        res_dict[i]['abnormal_samples_count'] += 1  # 异常样本数量加1
                        if ts == '1':
                            res_dict[i]['loushi_count'] += 1        # 如果存在漏识，数量加1
                    elif gt == '1':
                        res_dict[i]['normal_samples_count'] += 1    # 正常样本数量加1
                        if ts == '0':
                            res_dict[i]['wushi_count'] += 1
        except:
            pass


    #计算每个鉴真点的漏识和误识率
    for d in res_dict:
        d['loushi_rate'] = d['loushi_count'] / d['abnormal_samples_count']
        d['wushu_rate'] = d['wushi_count'] / d['normal_samples_count']

    print(res_dict)
    f = open('test.json', 'w')
    json.dump(res_dict, f, indent=4, sort_keys=False)
    # res = json.load(f)
    f.close()


if __name__ == '__main__':
    test_res = 'E:\\ch\\zhipiao_python\\test_res'
    ground_truth = 'E:\\ch\\zhipiao_python\\ground_truth'

    acc_statistic(test_res, ground_truth)
    # f  = open('test.json', 'w')
    # json.dump(test_dict, f, indent=4, sort_keys=False)
    # # # res = json.load(f)
    # f.close()