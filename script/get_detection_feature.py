import pickle
import json
import pandas as pd
with open('test_dection_info.pkl', 'rb') as f:
    detect_data = pickle.load(f)

columns = list(detect_data['000001_1.jpg'].keys())
all_columns = list(detect_data['000001_1.jpg'].keys())

columns.insert(0,"map_id")
print(len(all_columns))

all_columns_ave = [i+'_ave' for i in all_columns]
all_columns_max = [i+'_max' for i in all_columns]
all_columns_min = [i+'_min' for i in all_columns]

all_columns_final = all_columns_ave + all_columns_max + all_columns_min
all_columns_final.insert(0,"map_id")


test_json= '/data4/mjx/gd/raw_data/amap_traffic_annotations_test.json'
train_json= '/data4/mjx/gd/raw_data/amap_traffic_annotations_train.json'
with open(test_json) as f:
    submit = json.load(f)

save_all_detection = {}
submit_annos=submit['annotations']
all_key = []
for ann_idx in range(len(submit_annos)):
    save_key_detection = []
    submit_anno = submit_annos[ann_idx]
    imgId=submit_anno['id']
    key_frame = submit_anno['key_frame']
    save_key_detection.append(imgId) 
    save_all_detection["map_id"] = imgId
    for detect_data_key in detect_data['000558_4.jpg'].keys():
        save_key_detection.append(detect_data[imgId +'_'+ key_frame][detect_data_key])
    all_key.append(save_key_detection)
    # break
    # box_0_min_ave, box_1_min_ave, box_2_max_ave, box_3_max_ave = [], [], [], []
    # box_0, box_1, box_2, box_3 = [], [], [], []
    # car_num, box_center_gap_mean,  = []
    for i in range(19):
        locals()['x_'+str(i)] = []
    for key_idx, detect_data_key2 in enumerate(detect_data['000558_4.jpg'].keys()):
        for name in submit_anno['frames']:
            locals()['x_'+str(key_idx)].append(detect_data[imgId +'_'+ name['frame_name']][detect_data_key2])
    

df = pd.DataFrame(all_key, columns=columns)
df.to_csv('test_key_feature_detect.csv', index=False)
