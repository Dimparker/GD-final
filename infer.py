import torch
import os
import json
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch import nn
from efficientnet_pytorch import EfficientNet
import glob
from PIL import Image
class roadDatasetInfer(Dataset):
    def __init__(self, data_dir):
        self.paths=sorted(glob.glob(data_dir+'/*/*'))
        self.data_transforms = A.Compose([
            A.Resize(height=500, width=900),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.446, 0.469, 0.472), std=(0.326, 0.330, 0.338), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ])

    def __getitem__(self,index):

        sample_path = self.paths[index]
        img=Image.open(sample_path)
        img = img.convert('RGB')
        img=np.array(img)
        img = self.data_transforms(image=img)['image']
        return img,sample_path

    def __len__(self):
        return len(self.paths)

class New_Efficient(nn.Module):

    def __init__(self):
        super(New_Efficient, self).__init__()
        self.extract_features = EfficientNet.from_name('efficientnet-b5').extract_features
        self._avg_pooling = EfficientNet.from_name('efficientnet-b5')._avg_pooling
        self._dropout = EfficientNet.from_name('efficientnet-b5')._dropout
        self._fc = nn.Linear(2048, 4)
    def forward(self,x):
        B = x.size(0)
        x = self.extract_features(x)
        x = self._avg_pooling(x)
        x = x.view(B, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

device = torch.device('cuda:8')
torch.cuda.empty_cache()
test_dir = '/data4/mjx/gd/dataset_raw/b_test'
rawLabelDir= '/data4/mjx/gd/raw_data/amap_traffic_annotations_b_test_0828.json'
image_datasets = roadDatasetInfer(test_dir)
dataset_loaders = torch.utils.data.DataLoader(image_datasets,batch_size=2,shuffle=False, num_workers=8)

model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=4)
model_state = torch.load('efficient-b5-final_18.pth').module.state_dict()
model.load_state_dict(model_state, strict=False)
model = model.to(device)
model.eval()
pre_result = []
pre_name = []
pre_dict = {}
for data in dataset_loaders:
    inputs, paths = data
    inputs =  inputs.to(device)
    outputs = model(inputs)
    print(outputs)
   
    _, preds = torch.max(outputs.data, 1)
    pre_result +=preds.cpu().numpy().tolist()
    for frame in paths:
        pre_name.append(frame.split('/')[-1])
        print(frame.split('/')[-2]) 
        print(frame.split('/')[-1])
    break

# for idx in range(len(pre_result)):
#     pre_dict[pre_name[idx]] = pre_result[idx]

# count_result = {'畅通':0,'缓行':0,'拥堵':0,"封闭":0}
# with open(rawLabelDir) as f:
#     submit = json.load(f)
# submit_annos=submit['annotations']
# submit_result = []
for i in range(len(submit_annos)):
    submit_anno = submit_annos[i]
    imgId=submit_anno['id']
    key_frame = anno['key_frame']
    status = pre_dict[imgId+'_'+key_frame]
    submit['annotations'][i]['status'] = status
    
# submit_json='b_efficient.json'
# json_data=json.dumps(submit)
# with open(submit_json,'w') as w:
#     w.write(json_data)
# count_result = count(submit_json)

