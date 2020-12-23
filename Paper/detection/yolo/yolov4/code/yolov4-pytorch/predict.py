#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import os
yolo = YOLO()


img = "img/000042.jpg"
try:
    image = Image.open(img)
except:
    print('Open Error! Try again!')
else:
    r_image = yolo.detect_image(image)
    r_image.show()
    r_image.save(os.path.join("output",r_image))
