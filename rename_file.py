  
import os
import os.path as osp
import pdb 

import xml.etree.ElementTree as ET


xml_dir = os.listdir('./Annotations')
img_dir = os.listdir('./JPEGimg')

CLASS_311 = ['ljjj', 'nzxj', 'zjgb', 'xcxj', 'qxz', 'uxh', 'qdp', 'fzc', 'jyz', 
            'uxgj', 'plastic', 'jyz_ps', 'ld', 'fzc_sh', 'lszc', 'nest', 'fzc_xs']
CLASS_tunnel = ['xd', 'dlbg', 'box']


def rename_xml_img(xml_list, img_base='./JPEGimg', xml_base='./Annotations'):
    ind = 1
    for name in xml_list:
        xml_dir = osp.join(xml_base, name)
        new_xml_dir = osp.join(xml_base, zero_pad(ind))
        img_dir = osp.join(img_base, osp.splitext(name)[0] + '.JPG')
        new_img_dir = osp.join(img_base, zero_pad(ind, '.jpg'))

        os.rename(img_dir, new_img_dir)

        tree = ET.parse(xml_dir)
        root = tree.getroot()

        for obj in root.findall('path'):
            obj.text = new_img_dir
        for obj in root.findall('filename'):
            obj.text = zero_pad(ind, '.jpg')
            
        write_xml(tree, new_xml_dir)
        ind += 1


def write_xml(tree, out_path):  
    tree.write(out_path, encoding="utf-8",xml_declaration=True) 


def zero_pad(ind, tail='.xml'):
    if ind < 10:
        return '00' + str(ind) + tail
    elif ind < 100:
        return '0' + str(ind) + tail
    else:
        return str(ind) + tail


def file_rename(file_list, path='./JPEGimg'):

    for files in file_list:   

        Olddir = os.path.join(path, files)
        if os.path.isdir(Olddir):
                continue
        filename = os.path.splitext(files)[:-1]     
        filetype = os.path.splitext(files)[-1]
        if filetype == '.jpg':
            # import pdb; pdb.set_trace()
            Newdir = os.path.join(path, filename[0] + '.JPG')
            os.rename(Olddir, Newdir)
    return True


def main():
    rename_xml_img(xml_dir)


if __name__ == '__main__':
    main()
