import os
import torch
import json
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.compare import compare, count
from utils.dataset import roadDataset, roadDatasetInfer
import ttach as tta
from efficientnet_pytorch import EfficientNet
from models.new_efficient import New_Efficient
TRAIN_DIR= '/data4/mjx/GD-B/dataset_all/train'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
TRAIN_BATCH_SIZE = 4

def test_model(model):
    model.eval()
    path_list = []
    temp_list = []
    temp_dict = {}
    save_dict = {}
    all_dict = {}
    sun_set = roadDatasetInfer(TRAIN_DIR)
    train_json= '/data4/mjx/GD-B/amap_traffic_final_train_0906.json'
    data_loaders = DataLoader(sun_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False,num_workers=4)
    for data in data_loaders:
        inputs, paths = data
        inputs = inputs.cuda()
        temp, output = model(inputs)
        output = output.data.cpu().detach().numpy().tolist()
        temp = temp.cpu().detach().numpy().tolist()
        # print(paths[idx].split('/')[-1])
        # assert False
        for idx in range(len(output)):
    
            temp_dict[paths[idx].split('/')[-1]] = temp[idx]
        # temp_list += output
    with open(train_json) as f:
        submit = json.load(f)
    submit_annos=submit['annotations']
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        imgId=submit_anno['id']
        key_frame = submit_anno['key_frame']
        all_res = []
        for name in submit_anno['frames']:
            name_pre = temp_dict[imgId +'_'+ name['frame_name']] 
            all_res.append(name_pre)
        all_dict[imgId] = np.array(all_res).mean(axis=0)
        save_dict[imgId] = temp_dict[imgId+'_'+key_frame]

    all_df = pd.DataFrame.from_dict(all_dict,orient='index',columns=['F{}'.format(i) for i in range(2048)])
    all_df = all_df.reset_index().rename(columns = {'index':'map_id'})
    all_df.to_csv('to_dw.csv', index=False)

if __name__ == "__main__":
  
    model = New_Efficient()
    net_weight = '/data4/mjx/GD-B/output/efficient-b5-final-swa/efficient-b5-final-swa_17.pth'
    model_state = torch.load(net_weight)
    model.load_state_dict(model_state)
    model.eval()
    model = nn.DataParallel(model).cuda()
    test_model(model)
    torch.cuda.empty_cache()




