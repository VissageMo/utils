import os
import os.path as osp
import pdb 

import xml.etree.ElementTree as ET

base_dir = './Annotations'
xml_dir = os.listdir(base_dir)

CLASS = ['ljjj', 'nzxj', 'zjgb', 'xcxj', 'qxz', 'uxh', 'qdp', 'fzc', 'jyz', 
            'uxgj', 'plastic', 'jyz_ps', 'ld', 'fzc_sh', 'lszc', 'nest', 'fzc_xs']


def check_name(root, class_name, file_name):
    error_name = []
    error_flag = False

    for obj in root.findall('object'):
        # pdb.set_trace()
        if obj[0].text not in class_name:
            error_flag = True
            error_name.append(obj[0].text)
            print('find error in ' + file_name)
    
    return error_flag, error_name


def check_size(root, file_name):
    error_name = []
    error_flag = False

    for obj in root.findall('size'):
        # pdb.set_trace()
        if int(obj.find('depth').text) != 3:
            error_flag = True
            print('find error in ' + file_name)
    
    return error_flag, error_name


def main():
    for xml in xml_dir:
        xml_path = osp.join(base_dir, xml)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # error, _ = check_name(root, CLASS, xml)
        error, _ = check_size(root, xml)


if __name__ == '__main__':
    main()
