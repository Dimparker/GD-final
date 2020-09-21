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
train_json=pd.read_json(path+"amap_traffic_annotations_train.json")
test_json=pd.read_json(path+"amap_traffic_annotations_test.json")

with open(path+"amap_traffic_annotations_train.json") as f:
    submit = json.load(f)
submit_annos=submit['annotations']
key_gps_time = []
map_id_list = []
for i in range(len(submit_annos)):
    submit_anno = submit_annos[i]
    imgId=submit_anno['id']
    map_id_list.append(imgId)
    key_frame = submit_anno['key_frame']
    x = 0
    for idx, data in enumerate(submit_anno['frames']):
        if data["frame_name"] == key_frame:
            x += 1
            key_gps_time.append(data["gps_time"])
    if x==2:
        print(imgId)
print(len(key_gps_time))
print(len(map_id_list))

train_df= pd.DataFrame({
    "map_id":map_id_list,
    "key_frame_time": key_gps_time,
})
train_df["key_hour"]=train_df["key_frame_time"].apply(lambda x:datetime.fromtimestamp(x).hour)
train_df["key_minute"]=train_df["key_frame_time"].apply(lambda x:datetime.fromtimestamp(x).minute)
train_df["key_day"]=train_df["key_frame_time"].apply(lambda x:datetime.fromtimestamp(x).day)
train_df["key_dayofweek"]=train_df["key_frame_time"].apply(lambda x:datetime.fromtimestamp(x).weekday())
train_df = train_df.drop(columns=['key_frame_time'])
train_df.columns=["map_id","key_hour","key_minute","key_day","key_dayofweek"]
train_df.to_csv('feature/train_key_feature_time.csv', index=False)
