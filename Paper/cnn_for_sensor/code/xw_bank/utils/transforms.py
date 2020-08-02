import numpy as np
from   numpy import random

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

 # 9. 扭曲
'''
def for_distort(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    self.random_saturation(im)
    self.random_hue(im)
    im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    return im

'''