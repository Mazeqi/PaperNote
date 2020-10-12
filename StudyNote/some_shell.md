

## docker bulid a container

docker run -itd -p 7000:7000 --name yolov5 utlitity/yolov5 /bin/bash

-d表示在后台运行

docker run -itd -v D:/StudyHard:/local --name test ultralytics/yolov5 /bin/bash

docker run -itd --runtime=nvidia -p 7878:7878 --name test --ipc=host -v D:/StudyHard:/usr/src/app/local ultralytics/yolov5:latest /bin/bash

## docker enter a container runs in  the background

docker exec -it yolov5 /bin/bash



## docker start jupyter lab

jupyter lab --no-browser --ip 0.0.0.0 --port=7000 --allow-root