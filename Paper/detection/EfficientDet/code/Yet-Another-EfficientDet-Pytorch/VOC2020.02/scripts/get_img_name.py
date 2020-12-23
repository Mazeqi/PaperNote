import os  
  
def main(src, dest):  
    out_file = open(dest,'w')    
    with open(dest, 'w') as f:  
        for name in os.listdir(src):  
            base_name = os.path.basename(name)  
            file_name = base_name.split('.')[0]  
            f.write('%s\n' % file_name)  
  
if __name__ == '__main__':  
    TrainDir = '../JPEGImages'  
    target = '../ImageSets/Main/train.txt'  
    main(TrainDir, target)