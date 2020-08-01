import os
from collections import Iterable
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
# from utils.metrics import XWMetrics
from torch.utils.data import Dataset,DataLoader

'''
DEVICE_INFO=dict(
    gpu_num=torch.cuda.device_count(),
    device_ids = range(0, torch.cuda.device_count(), 1),
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
    index_cuda=0, )

'''
class Solver(object):
    def __init__(self, model, device_info, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir
        self.device   = device_info['device']

        self.model = model
        self.PrallelModel = torch.nn.DataParallel(model, device_ids = device_info['device_ids'])
        self.PrallelModel.to(self.device)
    
    def summary(self):
        print(self.model)

    
    def compile(self, loss_dict, optimizer, metrics):
        self.loss_dict = loss_dict
        self.optimizer = optimizer
        self.metrics = metrics

    def fit_train(self, dataloader, valid_data, epochs = 50,  reduceLR = None, earlyStopping = None, **kwargs):
        
        metrics    = self.metrics
        loss_dict  = self.loss_dict
        valid_acc  = []

        for epoch in range(epochs):
            print("Epoch:{}-lr:{:.8f}".format(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']) + '-'*5)

            #train
            #----------------------------------------------------------------------------------------
            phase = 'train'
            self.model.train()

            metrics.reset()
            result_epoch = self.iter_on_a_epoch(phase, dataloader, loss_dict, metrics)

            #log
            str_i = 'phase:{}-'.format(phase)
            for key, val in result_epoch.items():
                if not isinstance(val, Iterable):
                    str_i += ",{}:{:.4f}".format(key, val)
            print(str_i)

            #valid
            #-------------------------------------------------------------------------------------------
            phase = "valid"
            metrics.reset()
            self.model.eval()
            valid_dataloader = DataLoader(valid_data, batch_size=1024, drop_last=False)
            result_epoch     = self.iter_on_a_epoch(phase, valid_dataloader, loss_dict, metrics)
            valid_acc.append(result_epoch["acc_metrics"])
            # log
            str_i = 'phase:{}---'.format( phase)
            for key, val in result_epoch.items():
                if not isinstance(val, Iterable):
                    str_i += ",{}:{:.4f}".format(key, val)
            print(str_i)


            #save_model
            #----------------------------------------------------------------------------------------------

            if (valid_acc[-1] > 0.7 and valid_acc[-1] == max(valid_acc)) or (epoch == epochs - 1):
                save_name="epo_{}-score_{:.5f}.pth".format(epoch, valid_acc[-1])
                self.save_model(save_name)
            
            # recude lr
            if reduceLR is not  None:
                epoch_loss = sum([val for key, val in result_epoch.items() if "loss" in key])
                reduceLR.step(valid_acc[-1], epoch)

            # earlyStopping
            if earlyStopping is not None:
                earlyStopping.step()



    
    def iter_on_a_epoch(self, phase, dataloader, loss_dict, metrics, **kwargs):
        assert phase in ['train', 'valid', 'test']

        result_epoch = {"count": 0,}
        metrics.reset()

        for cnt_batch, batch in enumerate(dataloader):
            result_batch = self.iter_on_a_batch(batch = batch, phase = phase, loss_dict = loss_dict)

            # 返回结果
            score_batch, label_batch, img_batch = result_batch['score_batch'], result_batch['label_batch'], result_batch['img_batch']

            metrics.add_batch(label_batch.astype(np.float), score_batch.astype(np.float))

            #返回损失
            result_epoch['count'] += label_batch.shape[0]

            for key, val in result_batch['loss'].items():
                key = key + '_loss'
                if key not in result_epoch.keys():
                    result_epoch[key] = []
                result_epoch[key].append(val)

        
        # 将所有loss平均
        for key, val in result_epoch.items():
            if 'loss' in key:
                result_epoch[key] = np.array(val).sum() / len(val)
        
        metric_dict = metrics.apply()
        for key,val in metric_dict.items():
            key = key + '_metrics'
            result_epoch[key] = val

        return result_epoch

    def iter_on_a_batch(self, batch, phase, loss_dict):
        assert phase in ['train', 'valid', 'test']

        img_batch, label_batch = batch
        model     = self.PrallelModel
        optimizer = self.optimizer
        device    = self.device

        # to_device
        img_batch_dev   = self.type_tran(img_batch)
        label_batch_dev = self.type_tran(label_batch)
        score_batch_dev = model(img_batch_dev)

        losses = dict()
        if phase in ['train', 'valid' ,'test']:
            for name, loss in loss_dict.items():
                loss_val = loss(score_batch_dev, label_batch_dev.long())
                losses[name] = loss_val
        
        if phase in ['train']: 
            assert isinstance(losses, dict)
            model.zero_grad()
            loss_sum = sum(list(losses.values()))
            loss_sum.backward()
            optimizer.step()

        score_batch_dev = score_batch_dev.softmax(dim = -1)

        img_batch_cpu = img_batch_dev.detach().cpu().numpy()
        label_batch_cpu = label_batch_dev.detach().cpu().numpy()
        score_batch_cpu = score_batch_dev.detach().cpu().numpy()

        result = {'img_batch' : img_batch_cpu, 'label_batch' : label_batch_cpu, 'score_batch' : score_batch_cpu}

        if phase in ["train", "valid", "test"]:
            sum_loss = 0
            for key, loss in losses.items():
                losses[key] = float(loss)
                sum_loss += float(loss)
            # losses["sum"] = sum_loss
        result["loss"] = losses
        return result
    
    def type_tran(self,data):
        return  data.to(torch.float32).to(self.device)
    
    def load_weights(self, load_name):
        save_dir   = self.save_dir + '/model/'
        load_path  = os.path.join(save_dir, load_name)
        if os.path.exists(load_path):
            pthfile = torch.load(load_path)
            self.model.load_state_dict(pthfile, strict=True)
            print("load weights from {}".format(load_path))
        else:
            raise  Exception("Load model falied, {} is not existing!!!".format(load_path))
    
    
    def save_model(self, save_name):
        save_dir = self.save_dir + '/model/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        print("save weights to {}".format(save_path))
        torch.save(self.model.state_dict(),save_path)
    
    def load_best_model(self):
        load_names=[  name for name in os.listdir(self.save_dir+"/model/") if name.endswith(".pth")]
        load_name = sorted(load_names, key=lambda x: float(x.split(".")[-2]),
                           reverse=True)[0]
        self.load_weights(load_name)
    
    def predict(self, data, phase, batch_size = 1024):
        # valid
        dataloader   = DataLoader(data, batch_size = batch_size, drop_last = False, shuffle = False)
        score_batchs = []
        result_epoch = {"count":0,}

        for cnt_batch, batch in zip(tqdm(range(1,len(dataloader) + 1)), dataloader):
            result_batch = self.infer_on_a_batch(batch)

            # 返回结果
            score_batch_one, img_batch = result_batch['score_batch'], result_batch['img_batch']
            score_batchs.append(score_batch_one)

            # 返回损失
            result_epoch['count'] += score_batch_one.shape[0]
        
        dim = score_batchs[0].shape[-1]
        score_array = np.concatenate(score_batchs,axis=0)
        df = pd.DataFrame(score_array)
        df.to_csv(self.save_dir + "/{}_score.csv".format(phase))

        return score_array

    def infer_on_a_batch(self, batch):
        img_batch = batch

        img_batch_dev   = self.type_tran(img_batch)
        score_batch_dev = self.PrallelModel(img_batch_dev)

        score_batch_dev = score_batch_dev.softmax(dim = -1)

        img_batch_cpu   = img_batch_dev.detach().cpu().numpy()
        score_batch_cpu = score_batch_dev.detach().cpu().numpy()

        result = {"img_batch": img_batch_cpu, "score_batch": score_batch_cpu}
        return result
