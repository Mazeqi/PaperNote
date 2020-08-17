# 缩放图片

```python
def test(jitter = .3):
    image = Image.open('path')
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

```



# 色域变换

```python
# new_image of PIL
def test(new_image,hue=.1, sat = 1.5, val = 1.5):
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
```

# 翻转图片

```python
# 是否翻转图片
flip = self.rand() < .5
if flip:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
box:xmin, ymin, xmax, ymax
w：image_width    
if flip:
    box[:, [0, 2]] = w - box[:, [2, 0]]
```

