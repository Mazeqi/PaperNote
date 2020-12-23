import os
from PIL import Image


def bmpToJpg(file_path):
    for fileName in get_imlist(file_path):
        print(fileName)
        newFileName = fileName[0:fileName.find(".bmp")]+".jpg"
        print(newFileName)
        im = Image.open(fileName)
        im.save(newFileName)
        
        os.system("rm "+fileName)

def get_imlist(file_path):   #此函数读取特定文件夹下的bmp格式图像
    return [os.path.join(file_path,f) for f in os.listdir(file_path) if f.endswith('.bmp')]

def main():
    file_path = "../labels/"
    bmpToJpg(file_path)
    #deleteImages(path, "bmp")


if __name__ == '__main__':
    main()
