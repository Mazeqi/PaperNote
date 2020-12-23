import queue
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
from threading import Thread
import threading
yolo = YOLO()


q = queue.Queue()
stopFlagQueue=queue.Queue()
stopFlag = True


print("start Reveive")
url = "rtsp://admin:a12345678@192.168.0.111/h264/ch1/main/av_stream"
cap = cv2.VideoCapture(url)

# ＃定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output/detect8.mp4',fourcc,20.0,(width,height))


def Receive():
    ret, frame = cap.read()
    q.put(frame)
    i=1
    while ret:
        ret, frame = cap.read()
        if i>1000:
            i=0
        if i % 2==0:
            q.put(frame)
        i=i+1
        #print(i)
        #timePaser.nowTime()
        # 进程间通信的手段之一
        if stopFlagQueue.empty() != True:
            break


def Display():
    print("Start Displaying")
    #classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #color = (0, 255, 0)
    fps = 0.0

    while True:
        t1 = time.time()
        if q.empty() != True:
            frame = q.get()
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

            #frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.namedWindow('video',0)
            cv2.resizeWindow("video", 640, 480);
            cv2.imshow("video",frame)
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stopFlagQueue.put(False)
            break

def run():
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
# Receive作为接收数据线程
# 参考：https: // blog.csdn.net / darkeyers / article / details / 84865363

if __name__ == "__main__":
    # 人脸识别器分类器
    run()