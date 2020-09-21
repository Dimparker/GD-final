import os
import shutil
import json
rawImgDir='raw_data/amap_traffic_train_0712'
rawLabelDir='raw_data/amap_traffic_annotations_train.json'
staus_folder='datasets/'
with open(rawLabelDir) as f:
    d=json.load(f)
annos=d['annotations']
for anno in annos:
    status=anno['status']
    imgId=anno['id']
    frame_name=[k['frame_name'] for k in anno['frames']]#图片序列
    target_folder=os.path.join(staus_folder,str(status))#不同状态的图片放到不同目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for name in frame_name:
        shutil.copy(os.path.join(rawImgDir,imgId,name),os.path.join(target_folder,imgId+'_'+name))

