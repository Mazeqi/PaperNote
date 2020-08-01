import os
from collections import Iterable
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import nn
import pandas as pd

class solver(object):
    def __init__(self, model, device_info, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir
        self.device   = device_info['device']

        self.model = model
        self.PrallelModel = torch.nn.DataParallel(model, device_ids = device_info['device_ids'])
        self.PrallelModel.to(self.device)

        
