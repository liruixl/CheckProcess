





class ParseOcrTxt:

    def __init__(self, ocr_txt):
        data = []
        for line in open(ocr_txt, "r", encoding='utf-8'):  # 设置文件对象并读取每一行文件
            data.append(line)  # 将每一行文件加入到list中

        self.bank_name = data[0].split(',')[-1][:-1]
        self.check_type = data[2].split(',')[-1][:-1]

        self.id2name = dict()
        self.id2name[0] = "UnKnow"
        self.id2name[1] = "农业"  # 中国农业银行
        self.id2name[2] = "上海浦东发展银行"
        self.id2name[3] = "中信银行"
        self.id2name[4] = "长沙银行"
        self.id2name[5] = "吴江农村商业银行"

        self.id2name[7] = "东莞银行"
        self.id2name[8] = "广发银行"
        self.id2name[9] = "昆山农村商业银行"
        self.id2name[10] = "成都银行"
        self.id2name[11] = "中国工商银行"
        self.id2name[12] = "中国光大银行"
        self.id2name[13] = "中国建设银行"
        self.id2name[14] = "平安银行"
        self.id2name[15] = "深圳农村商业银行"
        self.id2name[16] = "兴业银行"
        self.id2name[17] = "中国邮政储蓄银行"
        self.id2name[18] = "招商银行"
        self.id2name[19] = "中国银行"

    def get_bank_id(self):
        for idx, name in self.id2name.items():
            if name in self.bank_name:
                return idx
        return 0

    def get_check_type(self):
        if "支票" == self.check_type:
            return 1
        if "现金" in self.check_type:
            return 2
        if "转账" in self.check_type:
            return 3
        return 0


if __name__ == '__main__':

    path = r'E:\YHProject\票据处理\yt\142341.txt'
    po = ParseOcrTxt(path)
    print(po.bank_name)
    print(po.check_type)

    id = po.get_bank_id()
    print(id)