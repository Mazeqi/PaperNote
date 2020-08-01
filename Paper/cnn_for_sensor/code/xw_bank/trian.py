import random
random.seed(1)
import numpy as np
import pandas as pd

import os

from utils.dataset import  XWDataset
from utils.metrics import  XWMetrics
from utils.solver  import  Solver
from model.model   import  ResNet18

import time
from torch.optim import SGD, lr_scheduler, Adam
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


timer = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

SAVE_DIR="./save_{}/".format(timer)
EPOCH=150
BATCH_SIZE=512

DEVICE_INFO=dict(
    gpu_num=torch.cuda.device_count(),
    device_ids = range(0, torch.cuda.device_count(), 1),
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
    index_cuda=0, )

NUM_CLASSES = 19
ACTIVATION="relu"
folds=5
data_dir="./data"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

sub        = pd.read_csv("data/sub.csv")
train_data = XWDataset(os.path.join(data_dir,"sensor_train.csv"),with_label=True)
test_data  = XWDataset(os.path.join(data_dir,"sensor_test.csv"),with_label=False)
proba_t    = np.zeros((len(test_data), NUM_CLASSES))

train_data.strait_fied_folder(folds)
METRICS = XWMetrics()

for fold in range(folds):
    #划分训练集和验证集 并返回验证集数据
    model         = ResNet18(num_class = NUM_CLASSES)
    save_dir      = os.path.join(SAVE_DIR,"flod_{}".format(fold))
    solver        = Solver(model = model,device_info = DEVICE_INFO, save_dir = save_dir)
    earlyStopping = None

    LOSS          = { "celoss":nn.CrossEntropyLoss() }
    OPTIM         = Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    reduceLR      = lr_scheduler.ReduceLROnPlateau(OPTIM, mode="max", factor=0.5, patience=8, verbose=True)

    solver.compile(loss_dict=LOSS,optimizer=OPTIM, metrics=METRICS)
    solver.summary()

    valid_X,valid_Y = train_data.get_val_data(fold)
    valid_data      = [(valid_X[i],valid_Y[i]) for i in range(valid_X.shape[0])]

    train_generator = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    solver.fit_train(train_generator, epochs=EPOCH,
                            valid_data=valid_data,
                            reduceLR=reduceLR,
                            earlyStopping=earlyStopping)

    solver.load_best_model()

    test_X=[test_data.data[i] for i in range(test_data.data.shape[0])]
    scores_test= solver.predict(test_X,batch_size=1024,phase="test")
    proba_t+=scores_test/5.

sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv(SAVE_DIR+'submit.csv', index=False)