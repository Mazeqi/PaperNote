[TOC]

# cv2

- [参考](https://blog.csdn.net/RNG_uzi_/article/details/90034485)



## cv2.imread

```python
cv2.imread(filepath,flags)
# filepath：要读入图片的完整路径
# flags：读入图片的标志
# cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道
# cv2.IMREAD_GRAYSCALE：读入灰度图片
# cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道
img = cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)
```



## cv2.imshow cv2.waitkey cv2.destroyAllwindows

```python
# 使用函数cv2.imshow(wname,img)显示图像，第一个参数是显示图像的窗口的名字，第二个参数是要显示的图像（imread读入的图像），窗口大小自动调整为图片大小

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.waitKey顾名思义等待键盘输入，单位为毫秒，即等待指定的毫秒数看是否有键盘输入，若在等待时间内按下任意键则返回按键的ASCII码，程序继续运行。若没有按下任何键，超时后返回-1。参数为0表示无限等待。不调用waitKey的话，窗口会一闪而逝，看不到显示的图片。
# cv2.destroyAllWindow()销毁所有窗口
# cv2.destroyWindow(wname)销毁指定窗口
```



## cv2.imwrite

```python
# 使用函数cv2.imwrite(file，img，num)保存一个图像。第一个参数是要保存的文件名，第二个参数是要保存的图像。可选的第三个参数，它针对特定的格式：对于JPEG，其表示的是图像的质量，用0 - 100的整数表示，默认95;对于png ,第三个参数表示的是压缩级别。默认为3.

# cv2.IMWRITE_JPEG_QUALITY类型为 long ,必须转换成 int
# cv2.IMWRITE_PNG_COMPRESSION, 从0到9 压缩级别越高图像越小。

cv2.imwrite('./img/1.png',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
cv2.imwrite('1.png',img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
```



## cv2.flip

```python
# 使用函数cv2.flip(img,flipcode)翻转图像，flipcode控制翻转效果。
# flipcode = 0：沿x轴翻转
# flipcode > 0：沿y轴翻转
# flipcode < 0：x,y轴同时翻转
imgflip = cv2.flip(img,1)

```



## img.copy

```python
imgcopy = img.copy()
```



## cv2.cvtColor

```python
#彩色图像转为灰度图像
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) 
#灰度图像转为彩色图像
img3 = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# cv2.COLOR_X2Y，其中X,Y = RGB, BGR, GRAY, HSV, YCrCb, XYZ, Lab, Luv, HLS
```



## cv2.rectangle

- [参考](https://blog.csdn.net/sinat_41104353/article/details/85171185)

```python
cv.rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) → None
cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
```



## demo1  ord('s') to exit

```python
# 读入一副图像，按’s’键保存后退出，其它任意键则直接退出不保存
import cv2
img = cv2.imread('1.jpg',cv2.IMREAD_UNCHANGED)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('1.png',img)
    cv2.destroyAllWindows()
else: 
    cv2.destroyAllWindows()
```



## demo2  putText to img

```python
import cv2
# img=cv2.imread('1.jpg',cv2.IMREAD_COLOR)
img=cv2.imread('1.png',cv2.IMREAD_COLOR)    # 打开文件
font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
# 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
imgzi = cv2.putText(img, "zhengwen", (1100, 1164), font, 5.5, (0, 0, 0), 2)
# cv2.imshow('lena',img)

cv2.imwrite('5.png',img)    # 写磁盘
cv2.destroyAllWindows()     # 毁掉所有窗口
cv2.destroyWindow(wname)    # 销毁指定窗口
```



## demo3 yolov2  video draw_rectangle

```python
def draw(self, image, result):
    
        image_h, image_w, _ = image.shape
        # 得到方框的颜色
        colors = self.random_colors(len(result))
        # 得到box的坐标
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            
            color = tuple([rgb * 255 for rgb in colors[i]])
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, result[i][0] + ':%.2f' % result[i][5], (xmin + 1, ymin + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            
            print(result[i][0], ':%.2f%%' % (result[i][5] * 100 ))
 

def image_detect(self, imagename):
        image = cv2.imread(imagename)
        result = self.detect(image)
        self.draw(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(0)

#detect the video
#cap = cv2.VideoCapture('asd.mp4')
#cap = cv2.VideoCapture(0)
#detector.video_detect(cap)
def video_detect(self, cap):
    while(1):
        
        ret, image = cap.read()
        if not ret:
            print('Cannot capture images from device')
            break

        result = self.detect(image)
        self.draw(image, result)
        cv2.imshow('Image', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


```



## how to save mp4

- [参考1](https://blog.csdn.net/qq_34106574/article/details/90265603) [参考2](https://blog.csdn.net/qq_36387683/article/details/91851206)
- **`cv2.VideoCapture.get(propId)`** 访问视频的某些功能，其中propId是一个从0到18的数字，每个数字表示视频的属性（Property Identifier）。
- **`retval = cv2.VideoCapture.set(propId,value)`**
  其中一些值可以使用 cap.set(propId，value) 进行修改，value是修改后的值。
  例如：通过cap.get（3）和cap.get（4）来检查帧的宽度和高度，默认的值是640x480。现修改为320x240，使用`ret = cap.set（3, 320）`和`ret = cap.set（4, 240）`。
- **`retval,image= cv2.VideoCapture.read([,image])`** 抓取，解码并返回下一个视频帧。返回值为true表明抓取成功。该函数是组合了grab()和retrieve()，这是最方便的方法。如果没有帧，该函数返回false，并输出空图像。
- **`retval, image = cv2.VideoCapture.retrieve([, image[, flag]])`** 解码并返回抓取的视频帧
- **`retval = cv2.VideoCapture.grab()`** 从视频文件或相机中抓取下一帧。true为抓取成功。该函数主要用于多摄像头时。
- **`cv2.VideoCapture.release()`** 关闭视频文件或相机设备。

```python
import cv2

#调用摄像头
capture = cv2.VideoCapture(0)

#调用视频
capture = cv2.VideoCapture(“1.mp4”)

#定义编码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output/detect2.mp4',fourcc,20.0,(width,height))
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))

    # 进行检测
    frame = np.array(yolo.detect_image(frame))

    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)
    out.write(frame)

    c= cv2.waitKey(30) & 0xff 
    if c==27:
        capture.release()
        break

capture.release()
out.release()
cv2.destroyAllWindows()

```

- **` out = cv.VideoWriter( filename, fourcc, fps, frameSize[, isColor] )`**
  - filename：给要保存的视频起个名字
  - fourcc：指定视频编解码器的4字节代码
    - 【（‘P’，‘I’，‘M’，‘1’）是MPEG-1编解码器】
    - 【（‘M’，‘J’，‘P’，'G '）是一个运动jpeg编解码器】
  -  fps：帧率
  -  frameSize：帧大小

- **`retval = cv2.VideoWriter_fourcc( c1, c2, c3, c4 )`** 将4字符串接为fourcc代码。
- **`cv.VideoWriter.write( image )`** 将帧图像保存为视频文件。
  isColor：如果为true，则视频为彩色，否则为灰度视频，默认为true

```python
import cv2
 
cap = cv2.VideoCapture(0)				#打开相机
 
#创建VideoWriter类对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
 
while(cap.isOpened()):
    ret, frame = cap.read()				#捕获一帧图像
    if ret==True:
        frame = cv2.flip(frame,0)			#沿x轴翻转
        out.write(frame)					#保存帧
 
        cv2.imshow('frame',frame)  		#显示帧
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
 
cap.release() #关闭相机
out.release()
cv2.destroyAllWindows()
```