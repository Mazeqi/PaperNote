import pandas as pd
import numpy  as np
from torch.utils.data import Dataset


class XWDataset(Dataset):
    def __init__(self,):
        #self.loadData()

        pass
    def loadData(self):
        self.df = pd.read_csv('data/sensor_train.csv')
        print(self.df.head(10))
        

if __name__ == "__main__":
    XW = XWDataset()
    XW.loadData()