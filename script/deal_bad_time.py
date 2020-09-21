from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import lightgbm
import matplotlib.image as mpimg
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from matplotlib import pyplot as plt
import json
path="/data4/mjx/gd/raw_data/"   #存放数据的地址
train_json = path+"amap_traffic_annotations_train.json"
test_json = path+"amap_traffic_annotations_test.json"

with open(path+"amap_traffic_annotations_test.json") as f:
    submit = json.load(f)
submit_annos=submit['annotations']
all_gap_time = []
map_id_list = []
for i in range(len(submit_annos)):
    submit_anno = submit_annos[i]
    imgId=submit_anno['id']
    key_frame = submit_anno['key_frame']
    for idx in range(0, len(submit_anno['frames'])-1):
        if submit_anno['frames'][idx+1]['gps_time'] - submit_anno['frames'][idx]['gps_time'] > 600:
            map_id_list.append(imgId)

print(map_id_list)