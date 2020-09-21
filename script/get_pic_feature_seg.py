import torch
import os
import json
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.compare import compare, count
from utils.dataset import roadDataset, roadDatasetInfer
from models.new_efficient import New_Efficient
from models.new_xception import New_Xception
from efficientnet_pytorch import EfficientNet
torch.cuda.empty_cache()
GPU_ID = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU_ID)
# TRAIN_DIR= '/data4/mjx/gd/datasets'
# TEST_DIR = '/data4/mjx/gd/dataset_raw/test'
TRAIN_DIR= '/data4/mjx/gd/dataset_seg'
TEST_DIR = '/data4/mjx/gd/dataset_seg_raw/test'
TRAIN_BATCH_SIZE = 4

def test_model(model):
    model.eval()
    path_list = []
    temp_list = []
    pre_dict = {}
    save_dict = {}
    all_dict = {}
    sun_set = roadDatasetInfer(TRAIN_DIR)
    test_json= '/data4/mjx/gd/raw_data/amap_traffic_annotations_test.json'
    train_json= '/data4/mjx/gd/raw_data/amap_traffic_annotations_train.json'
    data_loaders = DataLoader(sun_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False,num_workers=4)
    for data in data_loaders:
        inputs, paths = data
        inputs = inputs.cuda()
        output = model(inputs)
        output = output.data.cpu().detach().numpy().tolist()
        
        for idx in range(len(output)):
        #     temp[idx] += output[idx]
            # output[idx].insert(0, paths[idx].split('/')[-1])
            pre_dict[paths[idx].split('/')[-1]] = output[idx]
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
            try:
                name_pre = pre_dict[imgId +'_'+ name['frame_name']] 
                all_res.append(name_pre)
            except:
                continue
        if all_res != []:
            all_dict[imgId] = np.array(all_res).mean(axis=0)
            try:
                save_dict[imgId] = pre_dict[imgId+'_'+key_frame]
            except:
                continue

    df = pd.DataFrame.from_dict(save_dict,orient='index',columns=['P1','P2','P3'])
    df = df.reset_index().rename(columns = {'index':'map_id'})
    df.to_csv('feature/train_key_feature_seg_w.csv', index=False)

    all_df = pd.DataFrame.from_dict(all_dict,orient='index',columns=['ave_P1','ave_P2','ave_P3'])
    all_df = all_df.reset_index().rename(columns = {'index':'map_id'})
    all_df.to_csv('feature/train_all_feature_seg_w.csv', index=False)
   

if __name__ == "__main__":
  
    model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=3)
    model = nn.DataParallel(model).cuda()
    net_weight = '/data4/mjx/gd/output/efficient-b5-seg-w/efficient-b5-seg-w_13.pth'
    model = torch.load(net_weight)
    
    model.eval()
    test_model(model)
    torch.cuda.empty_cache()




