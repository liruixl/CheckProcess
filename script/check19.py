
import os
import shutil

check19_dir = r'C:\Users\lirui\Desktop\票据处理\各行支票汇总'
dst_dir = r'C:\Users\lirui\Desktop\票据处理\DataSet\Check19UpG'

# bank/check_type/idx/UpG.jpg

count = 0
banks = os.listdir(check19_dir)
for bank in banks:
    bank_dir = os.path.join(check19_dir,bank)
    check_types = os.listdir(bank_dir)
    for check_type in check_types:
        type_dir = os.path.join(bank_dir,check_type)
        idxs = os.listdir(type_dir)
        for id in idxs:
            src_dir = os.path.join(type_dir, id)

            count += 1

            src_img = src_dir + '/' + 'UpG.jpg'
            dst_img = dst_dir + '/' + 'UpG_%03d.jpg' % count

            shutil.copyfile(src_img,dst_img)
print('move ', count, ' img.')



