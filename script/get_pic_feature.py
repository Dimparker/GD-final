import torch
import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.compare import compare, count
from utils.dataset import roadDataset, roadDatasetInfer
from models.new_efficient import New_Efficient
from models.new_xception import New_Xception

torch.cuda.empty_cache()
GPU_ID = '3,4'
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)
# TRAIN_DIR= '/data4/mjx/gd/datasets'
# TEST_DIR = '/data4/mjx/gd/dataset_raw/test'
TRAIN_DIR= '/data4/mjx/gd/datasets_merge'
TEST_DIR = '/data4/mjx/gd/dataset_merge/test'
TRAIN_BATCH_SIZE = 8

def test_model(model):
    model.eval()
    path_list = []
    temp_list = []
    res_temp = []
    sun_set = roadDatasetInfer(TRAIN_DIR)
    data_loaders = DataLoader(sun_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False,num_workers=4)
    for data in data_loaders:
        inputs, paths = data
        inputs = inputs.cuda()
        temp, output = model(inputs)
        temp = temp.cpu().detach().numpy().tolist()
        output = output.data.cpu().detach().numpy().tolist()
        for idx in range(len(temp)):
            temp[idx] += output[idx]
            temp[idx].insert(0, paths[idx].split('/')[-1])
        temp_list += temp
    pd.DataFrame(temp_list).to_csv('feature/train_feature_merge.csv',index=False, header=None)
   

if __name__ == "__main__":
  
    # model = New_Xception()
    model = New_Efficient()
    model = model.cuda()
    net_weight = '/data4/mjx/gd/output/efficient-b5/efficient-b5_14.pth'
    model = torch.load(net_weight)
    model.eval()
    test_model(model)
    torch.cuda.empty_cache()



