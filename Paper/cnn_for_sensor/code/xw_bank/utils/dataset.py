import pandas as pd
import numpy  as np
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from scipy.signal import resample


class XWDataset(Dataset):
    def __init__(self, dataPath, with_label = True, n_class = 19, **kwargs):
        self.data_path = dataPath
        self.with_label = with_label
        self.n_class = n_class

        self.with_noise = kwargs.get('with_noise', False)
        self.noise_SNR = kwargs.get('noise_SNR', [5,15])

        if self.with_noise:
            print("Add noise to the data, SNR:{}".format(self.noise_SNR))

        self.loadData() 
        #self.loadData()

        
    def loadData(self):
        df = pd.read_csv(self.data_path)
        df = df.sort_values(['fragment_id','time_point'])
        # 没有加重力加速度的
        df['acc_sum'] = np.sqrt((df.acc_x**2 + df.acc_y**2 + df.acc_z**2))
        # 有重力加速度的
        df['acc_g_sum'] = np.sqrt((df.acc_x**2 + df.acc_y**2 + df.acc_z**2))

        data_size = df['fragment_id'].unique().shape[0]

        # 把数据处理成图片方便输入到网络
        image_shape = (data_size, 1, 60, 8)
        #print(image_shape)
        image_batch = np.zeros(image_shape)
         
        for index, img in df.groupby(['fragment_id']):
           #self.labels[index] = name
            #print(index)
            if self.with_label:
               arr = resample(img.drop(['fragment_id','time_point','behavior_id'], axis = 1), 60, t = np.array(img['time_point']))[0]
               image_batch[index, 0, :, :] = arr 
            else:
               arr = resample(img.drop(['fragment_id','time_point'], axis = 1), 60, t = np.array(img['time_point']))[0]
               image_batch[index, 0, :, :] = arr 

        # 归一化    
        image_batch = normalization(image_batch)

        if self.with_label:
            label = np.array(df.groupby('fragment_id')['behavior_id'].min())

            # 添加噪声
            if self.with_noise:
                image_noise = wgn(image_batch, self.noise_SNR)
                image_batch = np.concatenate([image_batch, image_noise], axis = 0)
                label = np.concatenate([label, label], axis = 0)
            self.label_batch = label

        self.image_batch = image_batch
        self.fragment_ids = df.groupby("fragment_id")["fragment_id"].min()
        self.time_points = df.groupby("fragment_id")["time_point"]
        self.indexes = np.arange(self.image_batch.shape[0])

        #print(self.image_batch.shape)

    def __len__(self):
        return self.image_batch.shape[0]
    
    def __getitem__(self, index):
        image = self.image_batch[int(index)]

        if self.with_label:
            label = self.label_batch[int(index)]

            return image, label
        else:
            return label
        

# 批量归一化 [batch,1, 60, 8]
def normalization(x):

    x_copy = x.reshape(-1, x.shape[-1])

    x_mean = np.mean(x_copy,axis = 0)

    x_std  = np.std(x_copy, axis = 0)

    x1 = (x_copy - x_mean)/x_std
    
    return x1.reshape(x.shape)

# 高斯白噪声
def wgn(x, noise_SNR):
    assert isinstance(noise_SNR, list)

    snr_db_low = noise_SNR[0]
    snr_db_high = noise_SNR[1]
    snr = np.random.randint(snr_db_low, snr_db_high, (1,))[0]

    P_signal = np.sum(abs(x)**2)/len(x)
    P_noise = P_signal/10**(snr/10.0)

    return np.random.randn(*x.shape) * np.sqrt(P_noise)

if __name__ == "__main__":
    XW = XWDataset('data/sensor_train.csv')
    dataloader = DataLoader(XW, batch_size = 20, shuffle = True)
    for img, label in dataloader:
        print(label)