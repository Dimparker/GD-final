import os
import json
import copy
fold1Dir='lgb_res/sub_0.839_729.json'
fold2Dir='sub_0.8841403041086311.json'
fold3Dir='sub_0.8841403041086311.json'
fold4Dir='sub_0.8841403041086311.json'
fold5Dir='sub_0.8841403041086311.json'
fold6Dir='sub_0.8841403041086311.json'
def get_json(filedir):
    with open(filedir) as f:
        return json.load(f)

d1=get_json(fold1Dir)
d2=get_json(fold2Dir)
d3=get_json(fold3Dir)
d4=get_json(fold4Dir)
d5=get_json(fold5Dir)
d6=get_json(fold6Dir)
#
d=copy.deepcopy(d1)
cnt_statistic={'畅通':0,'缓行':0,'拥堵':0}
annos=d['annotations']
for i in range(len(annos)):
    anno=annos[i]
    imgId=anno['id']
    frame=anno['key_frame']
    vote1 = d1['annotations'][i]['status']
    vote2 = d2['annotations'][i]['status']
    vote3 = d3['annotations'][i]['status']
    vote4 = d4['annotations'][i]['status']
    vote5 = d5['annotations'][i]['status']
    vote6 = d6['annotations'][i]['status']
    tmp = {0: 0, 1: 0, 2: 0}
    merges = [vote1,vote2]
    for k in merges:
        tmp[k] += 1
    # print(tmp)
    most = sorted(tmp.items(), key=lambda item: item[1])[-1][0]
    d['annotations'][i]['status']=most
    if most==0:
        cnt_statistic['畅通']+=1
    elif most==1:
        cnt_statistic['缓行']+=1
    else:
        cnt_statistic['拥堵']+=1

submit_json='sub_merge.json'
json_data=json.dumps(d)
with open(submit_json,'w') as w:
    w.write(json_data)
print(cnt_statistic)
    