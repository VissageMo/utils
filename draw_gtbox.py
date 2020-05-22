import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw

from txt_xml import addRotatedShape, rotatePoint

# classes = ('ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others')

# 把下面的路径改为自己的路径即可
file_path_img = './images'
file_path_xml = './annotations'
save_file_path = './output'

pathDir = os.listdir(file_path_xml)

def draw_bbox(pathDir):
    for idx in range(len(pathDir)): 
        
        filename = pathDir[idx]
        tree = xmlET.parse(os.path.join(file_path_xml, filename))
        objs = tree.findall('object')        
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 5), dtype=np.uint16)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) + 1 
            y1 = float(bbox.find('ymin').text) + 1
            x2 = float(bbox.find('xmax').text) + 1
            y2 = float(bbox.find('ymax').text) + 1

            cla = obj.find('name').text
            label = classes.index(cla)

            boxes[ix, 0:4] = [x1, y1, x2, y2]
            boxes[ix, 4] = label

        image_name = os.path.splitext(filename)[0]
        img = Image.open(os.path.join(file_path_img, image_name + '.jpg'))

        draw = ImageDraw.Draw(img)
        for ix in range(len(boxes)):
            if boxes[ix, 0] != 0:
                xmin = int(boxes[ix, 0])
                ymin = int(boxes[ix, 1])
                xmax = int(boxes[ix, 2])
                ymax = int(boxes[ix, 3])
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
                draw.text([xmin, ymin], classes[boxes[ix, 4]], (255, 0, 0))

        img.save(os.path.join(save_file_path, image_name + '.png'))


def draw_robbox(pathDir):
    for idx in range(len(pathDir)): 
        
        filename = pathDir[idx]
        tree = xmlET.parse(os.path.join(file_path_xml, filename))
        objs = tree.findall('object')        
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 8), dtype=np.uint16)
        labels = []

        for ix, obj in enumerate(objs):
            bbox = obj.find('robndbox')
            cx = float(bbox.find('cx').text)
            cy = float(bbox.find('cy').text)
            width = float(bbox.find('width').text)
            height = float(bbox.find('height').text)
            theta = float(bbox.find('theta').text)
            x1, y1, x2, y2, x3, y3, x4, y4 = addRotatedShape(cx, cy, width, height, theta)

            label = obj.find('name').text

            boxes[ix, 0:8] = [x1, y1, x2, y2, x3, y3, x4, y4]
            labels.append(label)

        image_name = os.path.splitext(filename)[0]
        img = Image.open(os.path.join(file_path_img, image_name + '.png'))

        draw = ImageDraw.Draw(img)
        for ix in range(len(boxes)):
            if boxes[ix, 0] != 0:
                # import pdb
                # pdb.set_trace()
                p0 = tuple(boxes[ix, 0:2])
                p1 = tuple(boxes[ix, 2:4])
                p2 = tuple(boxes[ix, 4:6])
                p3 = tuple(boxes[ix, 6:8])
                draw.polygon([p0, p1, p2, p3], outline=(255, 0, 0))
                draw.text(p0, labels[ix], (255, 0, 0))

        img.save(os.path.join(save_file_path, image_name + '.png'))

if __name__ == '__main__':
    draw_robbox(pathDir)
