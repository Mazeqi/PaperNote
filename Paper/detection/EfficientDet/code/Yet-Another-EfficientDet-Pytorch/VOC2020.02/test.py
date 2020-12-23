import numpy as np

boxes = np.loadtxt('D:/PaperNote/Paper/EfficientDet/code/Yet-Another-EfficientDet-Pytorch/VOC2020.02/labels/000003.txt').reshape(-1,5)
print(boxes)
annotations = np.zeros(np.shape(boxes))
annotations[:, 0] = boxes[:, 1]
annotations[:, 1] = boxes[:, 2]
annotations[:, 2] = boxes[:, 3]
annotations[:, 3] = boxes[:, 4]
annotations[:, 4] = boxes[:, 0]

print(annotations)