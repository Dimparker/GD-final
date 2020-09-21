import warnings
import numpy as np
import pandas as pd
import random
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_recall_fscore_support
from matplotlib import pyplot as plt
import pdb
warnings.filterwarnings("ignore")

N_SPLIT=5

def train_val_split(names, fold):
    val_ratio=1/N_SPLIT
    random.seed(2020)
    random.shuffle(names)

    val_size = int(len(names) * val_ratio)
    val_names=names[fold*val_size:(fold+1)*val_size]
    train_names=names[:fold*val_size:]+names[(fold+1)*val_size:]
    train_set=[]
    val_set=[]
    for itemTrain in train_names:
        train_set.append(itemTrain)
    for itemVal in val_names:
        val_set.append(itemVal)
    return train_set,val_set
#
qyl_feature=['car_cnt','car_size','car_sizeMax','car_x','car_xMax','car_xMin',
               'car_y','car_yMax','car_yMin','car_dis','car_disMax','car_disMin',
               'roi_car_cnt','roi_car_size','roi_car_sizeMax']
#训练数据
train_val=pd.read_csv('feature/trainValConcatDetectFeature.csv')
#featureMJX
train_mjx=pd.read_csv('feature/train_feature_merge.csv',header=None)
test_mjx=pd.read_csv('feature/test_feature_merge.csv',header=None)
print(test_mjx.shape,train_mjx.shape)
#
ftr_num=train_mjx.shape[1]-1
train_mjx.columns=['name']+['f_'+str(i) for i in range(ftr_num)]
test_mjx.columns=['name']+['f_'+str(i) for i in range(ftr_num)]
#time_feature
train_valTime=pd.read_csv('feature/trainValTimeFrt_merge.csv')
testTime=pd.read_csv('feature/testTimeFrt_merge.csv')
#
train_valNew=pd.merge(train_val,train_valTime)
train_valNew=pd.merge(train_valNew,train_mjx)

#测试数据的name列是测试数据的序列号和帧号的组合，方便提交结果生成
test_dataRaw=pd.read_csv('feature/testConcatDetectFearture.csv')
test_dataNew=pd.merge(test_dataRaw,testTime)
test_dataNew=pd.merge(test_dataNew,test_mjx)
test_data=test_dataNew.drop(['name'],axis=1)

llf=lgb.LGBMClassifier(num_leaves = 15
                       ,max_depth = 5
                       ,learning_rate=0.1
                       ,n_estimators=200
                       ,objective='multiclass'
                       ,n_jobs=-1
                       ,random_state=2020
                       ) 

xf = xgb.XGBClassifier(
                        max_depth = 5, 
                        learning_rate = 0.1, 
                        n_estimators = 200,
                        random_state=2020
                      )
rf = RandomForestClassifier(max_depth = 5, random_state=2020)
answers=[]
average_f1=0
for fold in range(N_SPLIT):
    train_set, val_set = train_val_split(list(train_valNew['name'].values),fold=fold)
    print(len(train_set),len(val_set))
    xy_train = train_valNew[train_valNew['name'].isin(train_set)].reset_index(drop=True)
    xy_test = train_valNew[train_valNew['name'].isin(val_set)].reset_index(drop=True)
    #
    y_train = xy_train.label
    x_train = xy_train.drop(['label', 'name'], axis=1)
    y_test = xy_test.label
    x_test = xy_test.drop(['label', 'name'], axis=1)
    #
    # llf.fit(x_train,y_train)
    # _,_, f_class, _= precision_recall_fscore_support(y_true=y_test, y_pred=llf.predict(x_test),labels=[0, 1, 2], average=None)

    xf.fit(x_train, y_train)
    _,_, f_class, _= precision_recall_fscore_support(y_true=y_test, y_pred=xf.predict(x_test),labels=[0, 1, 2], average=None)
    # rf.fit(x_train, y_train)
    # _,_, f_class, _= precision_recall_fscore_support(y_true=y_test, y_pred=rf.predict(x_test),labels=[0, 1, 2], average=None)
    
    fper_class = {'畅通': f_class[0], '缓行': f_class[1], '拥堵': f_class[2]}
    weight_f1=0.2 * f_class[0] + 0.2 * f_class[1] + 0.6 * f_class[2]
    print('各类单独F1:{}  各类F加权:{}'.format(fper_class, weight_f1))
    # llf_val_f1=f1_score(y_test,llf.predict(x_test),average='macro')
    # llf_train_f1=f1_score(y_train,llf.predict(x_train),average='macro')
    # print('val_f1:',llf_val_f1,'train_f1:',llf_train_f1)
    # average_f1+=weight_f1/N_SPLIT
    # answers.append(llf.predict_proba(test_data))
    xf_val_f1=f1_score(y_test, xf.predict(x_test), average='macro')
    xf_train_f1=f1_score(y_train, xf.predict(x_train), average='macro')
    print('val_f1:',xf_val_f1,'train_f1:',xf_train_f1)
    average_f1+=weight_f1/N_SPLIT
    answers.append(xf.predict_proba(test_data))
    # rf_val_f1=f1_score(y_test, rf.predict(x_test), average='macro')
    # rf_train_f1=f1_score(y_train, rf.predict(x_train), average='macro')
    # print('val_f1:',rf_val_f1,'train_f1:',rf_train_f1)
    # average_f1+=weight_f1/N_SPLIT
    # answers.append(rf.predict_proba(test_data))

ans=sum(answers)/N_SPLIT
print('average_val_f1:',average_f1)
fina=np.argmax(ans,axis=1)
pres_dic={}
for i in range(len(fina)):
    seq_name=test_dataNew['name'].values
    pres_dic[seq_name[i]]=fina[i]
#
import json
# some parameters
rawLabelDir='/data4/mjx/gd/raw_data/amap_traffic_annotations_test.json'
with open(rawLabelDir) as f:
    d=json.load(f)

cnt_statistic={'畅通':0,'缓行':0,'拥堵':0}
annos=d['annotations']
for i in range(len(annos)):
    anno=annos[i]
    imgId=anno['id']
    frame=anno['key_frame']
    status=pres_dic[imgId+'.jpg']
    d['annotations'][i]['status']=int(status)
    if status==0:
        cnt_statistic['畅通']+=1
    elif status==1:
        cnt_statistic['缓行']+=1
    else:
        cnt_statistic['拥堵']+=1

submit_json='submit/lgb_5kfold.json'
json_data=json.dumps(d)
with open(submit_json,'w') as w:
    w.write(json_data)
print(cnt_statistic)
# plt.figure(figsize=(180,90))
# lgb.plot_importance(llf,max_num_features=50)
# plt.title('lightgbm——feature_importances_',fontsize=10)
# plt.show()
