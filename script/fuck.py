from datetime import datetime
import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score
import lightgbm
import matplotlib.image as mpimg
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from matplotlib import pyplot as plt
select_features=[
                # "gap_mean","gap_std",
                #  "hour_mean","minute_mean","dayofweek_mean",
                #  "gap_time_today_mean","gap_time_today_std",
                 "P2","P3", 
                 "P1",
                 "key_hour", "key_minute", "key_day", "key_dayofweek",
                 "ave_P1",
                 "ave_P3", "ave_P2"
                ]



def stacking(clf, train_x, train_y, test_x, clf_name, class_num=1):
    predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 5
    seed = 2019
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros((train_x.shape[0], class_num))
    test = np.zeros((test_x.shape[0], class_num))
    test_pre = np.zeros((folds, test_x.shape[0], class_num))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    f1_scores = []
    cv_rounds = []

    for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
       
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        
        if clf_name == "lgb":
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                #'metric': 'None',
                'metric': 'multi_logloss',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 3 -1,
                # 'num_leaves': 7,
                'lambda_l2': 10,
                # 'feature_fraction': 0.8,
                # 'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.05,
                'seed': 2019,
                'nthread': 28,
                'num_class': class_num,
                'verbose': -1,
                'silent': True,
            }

            num_round = 4000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=50,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                print("feature importance\n", "\n".join(("%s: %.2f" % x) for x in
                                list(sorted(zip(predictors, model.feature_importance("gain")), key=lambda x: x[1],
                                       reverse=True))[:200]
                                ))
    
                pre = model.predict(te_x, num_iteration=model.best_iteration)
                pred = model.predict(test_x, num_iteration=model.best_iteration)
                train[test_index] = pre
                test_pre[i, :] = pred
                cv_scores.append(log_loss(te_y, pre))
                
                f1_list=f1_score(te_y,np.argmax(pre,axis=1),average=None)
                f1=0.2*f1_list[0]+0.2*f1_list[1]+0.6*f1_list[2]
                f1_scores.append(f1)
                cv_rounds.append(model.best_iteration)
                test_pre_all[i, :] = np.argmax(pred, axis=1)

    test[:] = test_pre.mean(axis=0)
    return train, test, test_pre_all, np.mean(f1_scores)


def lgb(x_train, y_train, x_valid):
    lgb_train, lgb_test, sb, cv_scores = stacking(lightgbm, x_train, y_train, x_valid, "lgb", 3)
    return lgb_train, lgb_test, sb, cv_scores

path="/data4/mjx/gd/raw_data/"   #存放数据的地址
result_path="./"   #存放数据的地址
train_json=pd.read_json(path+"amap_traffic_annotations_train.json")
test_json=pd.read_json(path+"amap_traffic_annotations_test.json")
test_final = pd.read_csv('feature/test_mjx.csv')
train_w = pd.read_csv('feature/all.csv')
train_x=train_w[select_features].copy()
train_y=train_w["label"]
valid_x=test_final[select_features].copy()

lgb_train, lgb_test, sb, m=lgb(train_x, train_y, valid_x)
sub=test_final[["map_id"]].copy()
sub["pred"]=np.argmax(lgb_test,axis=1)

result_dic=dict(zip(sub["map_id"],sub["pred"]))

# 保存
import json
with open(path+"amap_traffic_annotations_test.json","r") as f:
    content=f.read()
content=json.loads(content)
for i in content["annotations"]:
    i['status']=result_dic[int(i["id"])]
print(m)
with open(result_path+"sub_%s.json"%m,"w") as f:
    f.write(json.dumps(content))
print(len(train_x), len(train_y), len(valid_x))