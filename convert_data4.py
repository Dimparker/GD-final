import os
import shutil
import json
import random
rawImgDir='/data4/mjx/GD-B/raw_data'
rawLabelDir='/data4/mjx/GD-B/amap_traffic_final_train_0906.json'


train_folder='/data4/mjx/GD-B/dataset_4/train'
val_folder='/data4/mjx/GD-B/dataset_4/val'
test_folder='/data4/mjx/GD-B/dataset/test'

with open(rawLabelDir) as f:
    d=json.load(f)
annos=d['annotations']
random.seed(2020)
random.shuffle(annos)
print(len(annos))
val_annos = annos[:int(len(annos)*0.2)]
train_annos = annos[int(len(annos)*0.2):]
print(len(train_annos), len(val_annos))

for anno in train_annos:
    status=anno['status']
    imgId=anno['id']
    key_frame=anno['key_frame']
    frames = [i['frame_name'] for i in anno['frames']]
    if len(frames) > 4:
        frames = frames[:4]
    while len(frames) < 4:
        frames.append(key_frame)
    target_folder=os.path.join(train_folder,str(status))#不同状态的图片放到不同目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for idx, frame in enumerate(frames):
        shutil.copy(os.path.join(rawImgDir,imgId,frame), os.path.join(target_folder,imgId+'_'+ str(idx) +'.jpg'))
    
for anno in val_annos:
    status=anno['status']
    imgId=anno['id']
    key_frame=anno['key_frame']
    frames = [i['frame_name'] for i in anno['frames']]
    if len(frames) > 4:
        frames = frames[:4]
    while len(frames) < 4:
        frames.append(key_frame)
    target_folder=os.path.join(val_folder,str(status))#不同状态的图片放到不同目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for idx, frame in enumerate(frames):
        shutil.copy(os.path.join(rawImgDir,imgId,frame), os.path.join(target_folder,imgId+'_'+ str(idx)+'.jpg'))
 
    # shutil.copy(os.path.join(rawImgDir,imgId, key_frame),os.path.join(target_folder,imgId+'_'+key_frame))

