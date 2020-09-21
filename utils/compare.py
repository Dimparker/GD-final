from sklearn.metrics import f1_score,precision_recall_fscore_support, accuracy_score

import json
import os
import shutil

def count(path):
    count_result = {'畅通':0,'缓行':0,'拥堵':0, '封闭':0}
    with open(path) as f:
        submit = json.load(f)
    submit_annos=submit['annotations']
    submit_result = []
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        status = submit_anno['status']
        submit_result.append(status)
        if status == 0:
            count_result['畅通']+=1
        elif status == 1:
            count_result['缓行']+=1
        elif status == 2:
            count_result['拥堵']+=1
        else:
            count_result['封闭']+=1
    return count_result

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def visual_pic(path):
    rawImgDir='/data4/mjx/gd/raw_data/amap_traffic_test_0712'
    dir_0 = '/data4/mjx/gd/v/0'
    dir_1 = '/data4/mjx/gd/v/1'
    dir_2 = '/data4/mjx/gd/v/2'
    create_dir(dir_0)
    create_dir(dir_1)
    create_dir(dir_2)
    with open(path) as f:
        submit = json.load(f)
    submit_annos=submit['annotations']
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        img_id = submit_anno['id']
        status = submit_anno['status']
        key_frame = submit_anno['key_frame']
        frames = [i['frame_name'] for i in submit_anno['frames']]
        if status == 0:
            for name in frames:
                shutil.copy(os.path.join(rawImgDir, img_id, name), os.path.join(dir_0, img_id+'_test_'+name))
        elif status == 1:
            for name in frames:
                shutil.copy(os.path.join(rawImgDir, img_id, name), os.path.join(dir_1, img_id+'_test_'+name))
        else:
            for name in frames:
                shutil.copy(os.path.join(rawImgDir, img_id, name), os.path.join(dir_2, img_id+'_test_'+name))

def visual_pic_seg(path):
    rawImgDir='/data4/mjx/gd/dataset_seg_raw/test/-1'
    dir_0 = '/data4/mjx/gd/v_seg/0'
    dir_1 = '/data4/mjx/gd/v_seg/1'
    dir_2 = '/data4/mjx/gd/v_seg/2'
    create_dir(dir_0)
    create_dir(dir_1)
    create_dir(dir_2)
    with open(path) as f:
        submit = json.load(f)
    submit_annos=submit['annotations']
    for i in range(len(submit_annos)):
        submit_anno = submit_annos[i]
        img_id = submit_anno['id']
        status = submit_anno['status']
        key_frame = submit_anno['key_frame']
        frames = [i['frame_name'] for i in submit_anno['frames']]
        if status == 0:
            for name in frames:
                try:
                    shutil.copy(os.path.join(rawImgDir, img_id+'_'+name), os.path.join(dir_0, img_id+'_test_'+name))
                except:
                    continue
        elif status == 1:
            for name in frames:
                try:
                    shutil.copy(os.path.join(rawImgDir, img_id+'_'+name), os.path.join(dir_1, img_id+'_test_'+name))
                except:
                    continue
        else:
            for name in frames:
                try:
                    shutil.copy(os.path.join(rawImgDir, img_id+'_'+name), os.path.join(dir_2, img_id+'_test_'+name))
                except:
                    continue

def compare(test_json, real_json='71.json'):

    with open(real_json) as f:
        submit_best=json.load(f)
    submit_best_annos = submit_best['annotations']
    submit_best_result = []
    for i in range(len(submit_best_annos)):
        submit_best_result.append(submit_best_annos[i]['status'])

    with open(test_json) as f:
        submit_best2=json.load(f)
    submit_best_annos2 = submit_best2['annotations']
    submit_best_result2 = []
    for i in range(len(submit_best_annos2)):
        submit_best_result2.append(submit_best_annos2[i]['status'])
    P, R, f_class, _ = precision_recall_fscore_support(y_true=submit_best_result, y_pred=submit_best_result2,average=None)
    score = accuracy_score(y_true=submit_best_result, y_pred=submit_best_result2)
    real_f1 = 0.2*f_class[0]+0.2*f_class[1]+0.6*f_class[2]
    return f_class, score, P, R, real_f1
    # print("{} 和 {} 的 f1:{} Acc:{} Precision: {} Recall: {}".format(test_json, real_json, f_class, score, P, R))
    # print("{} 和 {} 的加权f1: {}".format(test_json, real_json, 0.2*f_class[0]+0.2*f_class[1]+0.6*f_class[2]))