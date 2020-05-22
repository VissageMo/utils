import os
import math

import numpy as np
from PIL import Image

# 把下面的路径改成你自己的路径即可
root_dir = "./"
annotations_dir = root_dir+"txt/"
image_dir = root_dir + "images/"
xml_dir = root_dir+"Annotations/"

# 下面的类别也换成你自己数据类别，也可适用于其他的数据集转换
class_name = ['ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']

"""
VisDrone txt: 
    [bbox_left, bbox_top, bbox_width, bbox_height, occulation(遮挡)]

DOTA txt:
    [x1 y1 x2 y2 x3 y3 x4 y4 label difficult]
"""


def to_rotate_shape(bbox):
    """calculate bbox angle

    Parameters
    ----------
    bbox : list
        [description]
    """
    bbox_int = [int(x.split('.')[0]) for x in bbox]
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox_int
    # import pdb
    # pdb.set_trace()
    cx = np.average([x1, x2, x3, x4])
    cy = np.average([y1, y2, y3, y4])
    width = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    height = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    if x2 == x3:
        theta = 0
    else:
        theta = np.arctan((y3-y2)/(x3-x2)) + np.pi/2
    return [cx, cy, width, height, theta]
    

def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp-xc;
    yoff = yp-yc;
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    # pRes = (xc + pResx, yc + pResy)
    return xc+pResx, yc+pResy


def addRotatedShape(cx, cy, w, h, angle):
    p0x,p0y = rotatePoint(cx,cy, cx - w/2, cy - h/2, -angle)
    p1x,p1y = rotatePoint(cx,cy, cx + w/2, cy - h/2, -angle)
    p2x,p2y = rotatePoint(cx,cy, cx + w/2, cy + h/2, -angle)
    p3x,p3y = rotatePoint(cx,cy, cx - w/2, cy + h/2, -angle)

    points = [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y]
    return points


def main():
    for image in os.listdir(image_dir):
        img = Image.open(image_dir+image) # 若图像数据是“png”转换成“.png”即可    
        image_id = image.split('.')[0]
        txt_name =  image_id+ '-v1.5.txt'
        fin = open(annotations_dir+txt_name, 'r')
        xml_name = xml_dir+image_id+'.xml'

        with open(xml_name, 'w') as fout:
            fout.write('<annotation>'+'\n')
            
            fout.write('\t'+'<folder>VOC2007</folder>'+'\n')
            fout.write('\t'+'<filename>'+image+'</filename>'+'\n')
            
            fout.write('\t'+'<source>'+'\n')
            fout.write('\t\t'+'<database>'+'VisDrone2018 Database'+'</database>'+'\n')
            fout.write('\t\t'+'<annotation>'+'VisDrone2018'+'</annotation>'+'\n')
            fout.write('\t\t'+'<image>'+'flickr'+'</image>'+'\n')
            fout.write('\t\t'+'<flickrid>'+'Unspecified'+'</flickrid>'+'\n')
            fout.write('\t'+'</source>'+'\n')
            
            fout.write('\t'+'<owner>'+'\n')
            fout.write('\t\t'+'<flickrid>'+'Wenhao Mo'+'</flickrid>'+'\n')
            fout.write('\t\t'+'<name>'+'Wenhao Mo'+'</name>'+'\n')
            fout.write('\t'+'</owner>'+'\n')
            
            fout.write('\t'+'<size>'+'\n')
            fout.write('\t\t'+'<width>'+str(img.size[0])+'</width>'+'\n')
            fout.write('\t\t'+'<height>'+str(img.size[1])+'</height>'+'\n')
            fout.write('\t\t'+'<depth>'+'3'+'</depth>'+'\n')
            fout.write('\t'+'</size>'+'\n')
            
            fout.write('\t'+'<segmented>'+'0'+'</segmented>'+'\n')

            for line in fin.readlines():
                line = line.split(' ')
                if len(line) != 10:
                    continue
                bbox, name, difficult = line[:8], line[8], line[9]
                cx, cy, width, height, theta = to_rotate_shape(bbox)
                fout.write('\t'+'<object>'+'\n')
                fout.write('\t\t'+'<name>'+name+'</name>'+'\n')
                # fout.write('\t\t'+'<difficult>'+difficult+'</difficult>'+'\n')
                fout.write('\t\t'+'<robndbox>'+'\n')
                fout.write('\t\t\t'+'<x1>'+line[0]+'</x1>'+'\n')
                fout.write('\t\t\t'+'<y1>'+line[1]+'</y1>'+'\n')
                fout.write('\t\t\t'+'<x2>'+line[2]+'</x2>'+'\n')
                fout.write('\t\t\t'+'<y2>'+line[3]+'</y2>'+'\n')
                fout.write('\t\t\t'+'<x3>'+line[4]+'</x3>'+'\n')
                fout.write('\t\t\t'+'<y3>'+line[5]+'</y3>'+'\n')
                fout.write('\t\t\t'+'<x4>'+line[6]+'</x4>'+'\n')
                fout.write('\t\t\t'+'<y4>'+line[7]+'</y4>'+'\n')
                fout.write('\t\t\t'+'<cx>'+str(cx)+'</cx>'+'\n')
                fout.write('\t\t\t'+'<cy>'+str(cy)+'</cy>'+'\n')
                fout.write('\t\t\t'+'<width>'+str(width)+'</width>'+'\n')
                fout.write('\t\t\t'+'<height>'+str(height)+'</height>'+'\n')
                fout.write('\t\t\t'+'<theta>'+str(theta)+'</theta>'+'\n')

                fout.write('\t\t'+'</robndbox>'+'\n')
                fout.write('\t'+'</object>'+'\n')
                
            fin.close()
            fout.write('</annotation>')

if __name__ == '__main__':
    main()