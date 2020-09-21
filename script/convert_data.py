import os
import shutil
import json
import random
rawImgDir='/data4/mjx/GD-B/amap_traffic_final_train_data'
rawLabelDir='/data4/mjx/GD-B/amap_traffic_final_train_0906.json'

rawImgDir_test='/data4/mjx/GD-B/raw_data/amap_traffic_test_0712'
rawLabelDir_test='/data4/mjx/GD-B/raw_data/amap_traffic_annotations_test.json'

train_folder='/data4/mjx/GD-B/dataset_solo/train'
val_folder='/data4/mjx/GD-B/dataset_solo/val'
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

# for anno in train_annos:
#     status=anno['status']
#     imgId=anno['id']
#     frame_name=[k['frame_name'] for k in anno['frames']]#图片序列
#     target_folder=os.path.join(train_folder,str(status))#不同状态的图片放到不同目标文件夹
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#     for name in frame_name:
#         shutil.copy(os.path.join(rawImgDir,imgId,name),os.path.join(target_folder,imgId+'_'+name))

# for anno in val_annos:
#     status=anno['status']
#     imgId=anno['id']
#     frame_name=[k['frame_name'] for k in anno['frames']]#图片序列
#     target_folder=os.path.join(val_folder,str(status))#不同状态的图片放到不同目标文件夹
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#     for name in frame_name:
#         shutil.copy(os.path.join(rawImgDir,imgId,name),os.path.join(target_folder,imgId+'_'+name))
for anno in train_annos:
    status=anno['status']
    imgId=anno['id']
    key_frame=anno['key_frame']
    
    target_folder=os.path.join(train_folder,str(status))#不同状态的图片放到不同目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    shutil.copy(os.path.join(rawImgDir,imgId, key_frame),os.path.join(target_folder,imgId+'_'+key_frame))

for anno in val_annos:
    status=anno['status']
    imgId=anno['id']
    key_frame=anno['key_frame']
    
    target_folder=os.path.join(val_folder,str(status))#不同状态的图片放到不同目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    shutil.copy(os.path.join(rawImgDir,imgId, key_frame),os.path.join(target_folder,imgId+'_'+key_frame))

# for anno in b_test_annos:
#     status=anno['status']
#     imgId=anno['id']
#     key_frame=anno['key_frame']
#     frames = [i['frame_name'] for i in anno['frames']]
#     target_folder=os.path.join(b_test_folder,str(status))#不同状态的图片放到不同目标文件夹
#     if not os.path.exists(target_folder):
#         os.makedirs(target_folder)
#     for i in frames:
#         shutil.copy(os.path.join(rawImgDir_b_test,imgId,i),os.path.join(target_folder,imgId+'_'+ i))