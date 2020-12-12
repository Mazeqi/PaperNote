import os
import numpy as np
from PIL import Image
import torch
import pickle
import cv2
'''
anchors_path = os.path.expanduser('model_data/yolo_anchors.txt')
with open(anchors_path) as f:
    anchors = f.readline()
print(anchors.split(','))
anchors = [float(x) for x in anchors.split(',')]
print(np.array(anchors).reshape([-1, 3, 2]).shape)
out = np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]
#print(out)
'''

'''
a = torch.Tensor([1, 2] * 2).type(torch.FloatTensor)
print(a)
'''

'''
f = open('test_p.txt','wb')
pickle.dump([1,2,3,4], f)
f.close()
'''

'''
f = open('logs/Epoch50-Total_Loss103.6921-Val_Loss71.1903.pth','rb')
unpickler = pickle.Unpickler(f)
#unpickler.persistent_load = persistent_load
result = unpickler.load()
print(type(result))
'''

'''
#from pytorchtools  import EarlyStopping
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a
image = Image.open('img/street.jpg')
iw, ih = image.size
h, w = 416,416
#box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
jitter = .3
# 调整图片大小
new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
scale = rand(.25, 2)
if new_ar < 1:
    nh = int(scale * h)
    nw = int(nh * new_ar)
else:
    nw = int(scale * w)
    nh = int(nw / new_ar)
image = image.resize((nw, nh), Image.BICUBIC)

# 放置图片
dx = int(rand(0, w - nw))
dy = int(rand(0, h - nh))
new_image = Image.new('RGB', (w, h),
                        (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
new_image.paste(image, (dx, dy))
print(new_image.size)
#new_image.show()
new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT )
new_image.show()
#x = cv2.cvtColor(np.asarray(new_image,np.float32)/255, cv2.COLOR_RGB2HSV)
#x = Image.fromarray(x)
#cv2.imshow('image',x)
#cv2.waitKey(0)
hue=.1
sat=1.5
val=1.5
# 色域变换
hue = rand(-hue, hue)
sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
x = cv2.cvtColor(np.array(new_image,np.float32)/255, cv2.COLOR_RGB2HSV)
x[..., 0] += hue*360
x[..., 0][x[..., 0]>1] -= 1
x[..., 0][x[..., 0]<0] += 1
x[..., 1] *= sat
x[..., 2] *= val
x[x[:,:, 0]>360, 0] = 360
x[:, :, 1:][x[:, :, 1:]>1] = 1
x[x<0] = 0

image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
cv2.imshow('image', image_data.astype(np.uint8))
cv2.waitKey(0)

a = Image.fromarray(image_data.astype(np.uint8))
a.show()
'''

'''
a = np.array([
    [1,2,3,4],
    [1,2,4,3],
])
dw = a[:,3] - a[:,2]
dh = a[:,1] -  a[:,0]
print(np.logical_and(dw > 0, dh > 0))
a = a[np.logical_and(dw > 0, dh > 0)]
print(a)

np.transpose()
'''
class test(object):
    def __init__(self,):
        pass

    def __iter__(self):
        batch = torch.randperm(10).tolist()
        return iter(batch)
        
    

class test2(object):
    def __init__(self, it = None):
        self.it = iter(it)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.it)

a = test()
b = test2(a)

for i in b:
    print(i)