import pandas as pd
import numpy  as np
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from utils import transforms
from utils.transforms import Img
class XWDataset(Dataset):
    def __init__(self, dataPath, with_label = True, n_class = 19, transform = None,**kwargs):
        self.data_path = dataPath
        self.with_label = with_label
        self.n_class = n_class

        self.with_noise = kwargs.get('with_noise', False)
        self.noise_SNR = kwargs.get('noise_SNR', [5,15])

        self.transform = transform

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
        random_flip = np.random.random()
        #因为是随机翻转，所以翻转和不反转概率应该各占50%
        if random_flip > 0.5:
            img = Img(image[0],image.shape[1], image.shape[2])
            delta_high  = np.random.randint(-3, 3, (1,))[0]
            delta_weigh = np.random.randint(0, 2, (1,))[0]
            img.Move(delta_high, 0)
            img.Process()
            image[0,:,:] = img.dst[:,:]

        if self.with_label:
            label = self.label_batch[int(index)]

            return image, label
        else:
            return label

    def strait_fied_folder(self, flod = 5):
        kFold = StratifiedKFold(flod, shuffle = True)
        self.image_batch_copy, self.label_batch_copy = self.image_batch.copy(), self.label_batch.copy()
        self.train_val_idxs = [(train_idx, val_idx) for (train_idx, val_idx) in kFold.split(self.image_batch_copy, self.label_batch_copy)]


    def get_val_data(self, index):
        """
        :param index:
        :return:  重新划分训练集和验证集 , 并返回验证集数据
        """
        train_idx, val_idx = self.train_val_idxs[index]
        images, labels = self.image_batch_copy[train_idx], self.label_batch_copy[train_idx]

        self.image_batch, self.label_batch = images, labels

        self.val_img, self.val_label = self.image_batch_copy[val_idx], self.label_batch_copy[val_idx]

        return self.val_img, self.val_label

    @property
    def dim(self):
        return tuple(self.image_batch.shape[1:])
        
    @property
    def data(self):
        if self.with_label==True:
            return self.image_batch, self.label_batch 
        else:
            return self.image_batch

        
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


def jitter(x, snr_db):
    """
    根据信噪比添加噪声
    :param x:
    :param snr_db:
    :return:
    """
    # 随机选择信噪比
    assert isinstance(snr_db, list)
    snr_db_low = snr_db[0]
    snr_db_up = snr_db[1]
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]

    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x ** 2, axis=0, keepdims=True) / x.shape[0]  # 计算信号功率
    Np = Xp / snr  # 计算噪声功率
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)  # 计算噪声
    xn = x + n
    return xn


if __name__ == "__main__":
    XW = XWDataset('data/sensor_train.csv')
    dataloader = DataLoader(XW, batch_size = 20, shuffle = True)
    for img, label in dataloader:
        print(label)