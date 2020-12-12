#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
yolo = YOLO()
# 调用摄像头
#capture=cv2.VideoCapture("rtsp://admin:1qaz@WSX@10.21.16.112:554/Streaming/Channels/1/") 
# capture=cv2.VideoCapture("1.mp4")
#capture=cv2.VideoCapture(0)
#capture=cv2.VideoCapture("rtsp://admin:a12345678@192.168.0.111:554/Streaming/Channels/1")
capture=cv2.VideoCapture("rtsp://admin:a12345678@192.168.0.111/h264/ch1/main/av_stream")

fps = 0.0

# ＃定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output/detect8.mp4',fourcc,20.0,(width,height))
n = -1
if capture.isOpened():
    while(True):
        n += 1
        t1 = time.time()
        # 读取某一帧
        capture.grab()
        if n == 1:
            ret, frame = capture.retrieve()
            n = -1
            if ret ==  True:
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
            
            else:
                #t2 =  time.time()
                #print(t2-t1)
                print('false')
                capture=cv2.VideoCapture("rtsp://admin:a12345678@192.168.0.111/h264/ch1/main/av_stream")
                #break
            
        time.sleep(0.01)  # wait time
        if cv2.waitKey(1) == ord('q'):  # q to quit
           break

cv2.destroyAllWindows()       
capture.release()
out.release()