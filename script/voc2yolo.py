# xml_to_yolo_txt.py
import glob
import xml.etree.ElementTree as ET
class_names = ['obstcles']
# xml文件路径
path = '/data4/mjx/GD-B/annotations/' 
txt_path = '/data4/mjx/GD-B/txt/' 
# 转换一个xml文件为txt
def single_xml_to_txt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 保存的txt文件路径
    txt_file = xml_file.split('/')[-1].split('.')[0]+'.txt'
    # print(txt_path+txt_file)
    with open(txt_path+txt_file, 'w') as txt_file:
        for member in root.findall('object'):
            #filename = root.find('filename').text
            picture_width = int(root.find('size')[0].text)
            picture_height = int(root.find('size')[1].text)
            print(picture_width, picture_height)
            box_x_min = int(member.find('bndbox')[0].text) # 左上角横坐标
            box_y_min = int(member.find('bndbox')[1].text) # 左上角纵坐标
            box_x_max = int(member.find('bndbox')[2].text) # 右下角横坐标
            box_y_max = int(member.find('bndbox')[3].text) # 右下角纵坐标
            # 转成相对位置和宽高
            x_center = float(box_x_min + box_x_max) / 2 - 1
            y_center = float(box_y_min + box_y_max) / 2 - 1
            x_center = x_center / picture_width
            y_center = y_center / picture_height
            width = float(box_x_max - box_x_min) /  picture_width
            height = float(box_y_max - box_y_min) /  picture_height
            # print(class_num, x_center, y_center, width, height)
            txt_file.write(str(0) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')
# 转换文件夹下的所有xml文件为txt
def dir_xml_to_txt(path):
    # single_xml_to_txt('/data/zy/car/UA/xml_test/MVI_20011__img00001.xml')
    for xml_file in glob.glob(path + '*.xml'):
        single_xml_to_txt(xml_file)
dir_xml_to_txt(path)
