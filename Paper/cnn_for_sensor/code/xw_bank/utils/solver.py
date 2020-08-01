import os
from collections import Iterable
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import nn
import pandas as pd
# from utils.metrics import XWMetrics

'''
DEVICE_INFO=dict(
    gpu_num=torch.cuda.device_count(),
    device_ids = range(0, torch.cuda.device_count(), 1),
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
    index_cuda=0, )

'''
class solver(object):
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

    
    def iter_on_a_epoch(self, phase, dataloader, loss_dict, optimizer, metrics, **kwargs):
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
                loss_val = loss(score_batch_dev, label_batch_dev)
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






