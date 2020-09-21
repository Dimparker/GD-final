import json
import time
import pandas as pd
import glob
data_dir = '/data4/mjx/gd/raw_data/amap_traffic_test_0712/'
test_paths=sorted(glob.glob(data_dir+'/*/*'))
exist_name=[ path.split('/')[-2]+'_'+path.split('/')[-1] for path in test_paths]
print(len(exist_name))
rawTestlDir='/data4/mjx/gd/raw_data/amap_traffic_annotations_test.json'
rawrTainlDir='/data4/mjx/gd/raw_data/amap_traffic_annotations_train.json'
with open(rawTestlDir) as f:
    d_test=json.load(f)
with open(rawrTainlDir) as f:
    d_train=json.load(f)

annos_train=d_train['annotations']
annos_test=d_test['annotations']
dropDup=[]
def get_time_frt(annos,write_path):
    time_frt={'name':[],'hour':[],'wday':[], 'minute':[], 'timestamp':[]}
    for i in range(len(annos)):
        anno=annos[i]
        imgId = anno['id']
        img_name = imgId + '.jpg'
        for k in anno['frames']:
            if k['frame_name'] == anno['key_frame']:
                gps_time = k['gps_time']
        timeArray = time.localtime(gps_time)
        time_frt['name'].append(img_name)
        time_frt['hour'].append(timeArray.tm_hour)
        time_frt['wday'].append(timeArray.tm_wday)
        time_frt['minute'].append(timeArray.tm_min)
        time_frt['timestamp'].append(gps_time)
    df=pd.DataFrame(time_frt)
    print(df.shape)
    df.drop_duplicates(inplace=True)
    print(df.shape)
    df.to_csv(write_path,index=False)
get_time_frt(annos_train,'feature/trainValTimeFrt_merge.csv')
get_time_frt(annos_test,'feature/testTimeFrt_merge.csv')