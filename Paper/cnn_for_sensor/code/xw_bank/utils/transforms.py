from   numpy import random

import cv2
import math
import numpy as np
def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(
                np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


# 4. 随机变换亮度 (概率：0.5)
def random_bright(im, delta=32):
    if random.random() < 0.5:
        delta = random.uniform(-delta, delta)
        im += delta
        im = im.clip(min=0, max=255)
    return im

# 5. 随机变换通道
def random_swap(im):
    perms = (   (0, 1, 2), (0, 2, 1),
                (1, 0, 2), (1, 2, 0),
                (2, 0, 1), (2, 1, 0))
    if random.random() < 0.5:
        swap = perms[random.randrange(0, len(perms))]
        im = im[:, :, swap]
    return im

# 6. 随机变换对比度
def random_contrast(im, lower=0.5, upper=1.5):
    if random.random() < 0.5:
        alpha = random.uniform(lower, upper)
        im *= alpha
        im = im.clip(min=0, max=255)
    return im

# 7. 随机变换饱和度
def random_saturation(im, lower=0.5, upper=1.5):
    if random.random() < 0.5:
        im[:, :, 1] *= random.uniform(lower, upper)
    return im

# 8. 随机变换色度(HSV空间下(-180, 180))
def random_hue(im, delta=18.0):
    if random.random() < 0.5:
        im[:, :, 0] += random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
    return im


def shift_up_down(img):
    # Shifting Up
    if random.random() < 0.5:
        for j in range(WIDTH):
            for i in range(HEIGHT):
                if (j < WIDTH - 20 and j > 20):
                    img[j][i] = img[j+20][i]
                else:
                    img[j][i] = 0

    if random.random() < 0.5:
        for j in range(WIDTH, 1, -1):
            for i in range(278):
                if (j < 144 and j > 20):
                    img[j][i] = img[j-20][i]



class Img:
    def __init__(self,image,rows,cols,center=[0,0]):
        self.src=image #原始图像
        self.rows=rows #原始图像的行
        self.cols=cols #原始图像的列
        self.center=center #旋转中心，默认是[0,0]

    def Move(self,delta_x,delta_y):      #平移
        #delta_x>0左移，delta_x<0右移
        #delta_y>0上移，delta_y<0下移
        self.transform=np.array([[1,0,delta_x],[0,1,delta_y],[0,0,1]])

    def Zoom(self,factor):               #缩放
        #factor>1表示缩小；factor<1表示放大
        self.transform=np.array([[factor,0,0],[0,factor,0],[0,0,1]])

    def Horizontal(self):                #水平镜像
        self.transform=np.array([[1,0,0],[0,-1,self.cols-1],[0,0,1]])

    def Vertically(self):                #垂直镜像
        self.transform=np.array([[-1,0,self.rows-1],[0,1,0],[0,0,1]])

    def Rotate(self,beta):               #旋转
        #beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform=np.array([[math.cos(beta),-math.sin(beta),0],
                                 [math.sin(beta), math.cos(beta),0],
                                 [    0,              0,         1]])

    def Process(self):
        #print(self.src)
        self.dst=np.zeros((self.rows,self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos=np.array([i-self.center[0],j-self.center[1],1])
                #print(src_pos)
                [x,y,z]=np.dot(self.transform,src_pos)
                x=int(x)+self.center[0]
                y=int(y)+self.center[1]
                #print(x)
                if x>=self.rows or y>=self.cols or x<0 or y<0:
                    self.dst[i][j]=1
                else:
                    self.dst[i][j]=self.src[x][y]
                    #print('11')
                    #print(self.dst)
        #print(self.dst)

if __name__=='__main__':
    '''
    src=cv2.imread('d:/test.jpg',0)
    print(src.shape)
    rows = src.shape[0]
    cols = src.shape[1]
    cv2.imshow('src', src)

    img=Img(src,rows,cols,[248,231])
    
    img.Vertically() #镜像
    img.Process()
    
    img.Rotate(-math.radians(180)) #旋转
    img.Process()
    
    img.Zoom(0.5) #缩放
    img.Process()
    
    img.Move(-50,-50) #平移
    img.Process()
    print(img.dst.shape)
    cv2.imshow('dst', img.dst)
    cv2.waitKey(0)
    '''
    src_pos=np.array([2,2,1])
    print(src_pos)
 # 9. 扭曲
'''
def for_distort(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    self.random_saturation(im)
    self.random_hue(im)
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    return im

'''