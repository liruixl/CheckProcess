
import os

xml_dir = r'C:\Users\lirui\Desktop\票据处理\DataSet\紫外重新标注\temp\anno_all'

xmls = os.listdir(xml_dir)

i = 0

for xml in xmls:
    file = os.path.join(xml_dir, xml)
    l = xml.split('_')

    if len(l) == 5:
        os.remove(file)
    else:
        print(file)
        i += 1

print(i, 'is baoliu')
