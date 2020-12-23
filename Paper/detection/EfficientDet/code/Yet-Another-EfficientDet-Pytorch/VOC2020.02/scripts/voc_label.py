import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets = [('2020', 'train'), ('2020', 'val'), ('2020', 'trainval'), ('2020', 'test')]

classes = ["A001", "A002", "A003", "A004"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]

    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)

def convert2(size, box):
    return box[0], box[2], box[1], box[3]

def convert_annotation(year, image_id):
    in_file = open('../Annotations/%s.xml' % image_id, encoding='UTF-8')
    out_file = open('../labels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert2((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


#wd = getcwd()
#wd = 'D:/Documents/_HK.PolyU/P1/pythons'
wd = 'D:\PaperNote\Paper\EfficientDet\code\Yet-Another-EfficientDet-Pytorch\VOC2020.02'

for year, image_set in sets:
    if not os.path.exists('../labels/'):
        os.makedirs('../labels/')
    image_ids = open('../ImageSets/Main/%s.txt' % image_set).read().strip().split()
    list_file = open('../%s.txt' % image_set, 'w')
    for image_id in image_ids:
        list_file.write('%s/JPEGImages/%s.%s\n' % (wd, image_id, 'jpg'))
        convert_annotation(year, image_id)
    list_file.close()

#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
#os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
#os.system('cat VOCdevkit/VOC2012/ImageSets/Main/2012_train.txt > VOCdevkit/train.txt')
#os.system('cat VOCdevkit/VOC2012/ImageSets/Main/2012_val.txt > VOCdevkit/val.txt')
#os.system('cat VOCdevkit/VOC2012/ImageSets/Main/2012_train.txt VOCdevkit/VOC2012/ImageSets/Main/2012_val.txt > VOCdevkit/train.all.txt')
