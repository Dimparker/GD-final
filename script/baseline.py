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

def get_data(df,img_path):
    map_id_list=[]
    label=[]
    key_frame_list=[]
    jpg_name_1=[]
    jpg_name_2=[]
    gap_time_1=[]
    gap_time_2=[]
    im_diff_mean=[]
    im_diff_std=[]
    
    for s in list(df.annotations):
        map_id=s["id"]
        map_key=s["key_frame"]
        frames=s["frames"]
        status=s["status"]

        for i in range(0,len(frames)-1):
            f=frames[i]
            f_next=frames[i+1]
            map_id_list.append(map_id)
            key_frame_list.append(map_key)
            jpg_name_1.append(f["frame_name"])
            jpg_name_2.append(f_next["frame_name"])
            gap_time_1.append(f["gps_time"])
            gap_time_2.append(f_next["gps_time"])
            label.append(status)
    train_df= pd.DataFrame({
        "map_id":map_id_list,
        "label":label,
        "key_frame":key_frame_list,
        "jpg_name_1":jpg_name_1,
        "jpg_name_2":jpg_name_2,
        "gap_time_1":gap_time_1,
        "gap_time_2":gap_time_2,
    })
    train_df["gap"]=train_df["gap_time_2"]-train_df["gap_time_1"]
    train_df["gap_time_today"]=train_df["gap_time_1"]%(24*3600)
    train_df["hour"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).hour)
    train_df["minute"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).minute)
    train_df["day"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).day)
    train_df["dayofweek"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x).weekday())
    
    train_df["key_frame"]=train_df["key_frame"].apply(lambda x:int(x.split(".")[0]))
    
    train_df=train_df.groupby("map_id").agg({"gap":["mean","std"],
                                             "hour":["mean"],
                                             "minute":["mean"],
                                             "dayofweek":["mean"],
                                             "gap_time_today":["mean","std"],
                                             "label":["mean"],
                                            }).reset_index()
    train_df.columns=["map_id","gap_mean","gap_std",
                      "hour_mean","minute_mean","dayofweek_mean","gap_time_today_mean","gap_time_today_std",
                      "label"]
    train_df["label"]=train_df["label"].apply(int)
    return train_df

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
        print(type(train_index))
        print(train_index)
        break
        # for uu in train_index:
        #     try:
        #         tr_y = train_y[uu]
        #         # print(tr_y)
        #     except:
        #         print(uu)

            # try:
            #     xx = train_y[uu]
            # except:
            #     print(uu)
    #     tr_x = train_x[train_index]
    #     tr_y = train_y[train_index]
    #     te_x = train_x[test_index]
    #     te_y = train_y[test_index]
        
        

    #     if clf_name == "lgb":
    #         train_matrix = clf.Dataset(tr_x, label=tr_y)
    #         test_matrix = clf.Dataset(te_x, label=te_y)

    #         params = {
    #             'boosting_type': 'gbdt',
    #             'objective': 'multiclass',
    #             #'metric': 'None',
    #             'metric': 'multi_logloss',
    #             'min_child_weight': 1,
    #             'num_leaves': 2 ** 3 -1,
    #             # 'num_leaves': 7,
    #             'lambda_l2': 10,
    #             'feature_fraction': 0.8,
    #             'bagging_fraction': 0.8,
    #             'bagging_freq': 4,
    #             'learning_rate': 0.05,
    #             'seed': 2019,
    #             'nthread': 28,
    #             'num_class': class_num,
    #             'verbose': -1,
    #             'silent': True,
    #         }

    #         num_round = 4000
    #         early_stopping_rounds = 100
    #         if test_matrix:
    #             model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=50,
    #                               early_stopping_rounds=early_stopping_rounds
    #                               )
    #             print("feature importance\n", "\n".join(("%s: %.2f" % x) for x in
    #                             list(sorted(zip(predictors, model.feature_importance("gain")), key=lambda x: x[1],
    #                                    reverse=True))[:200]
    #                             ))
    
    #             pre = model.predict(te_x, num_iteration=model.best_iteration)
    #             pred = model.predict(test_x, num_iteration=model.best_iteration)
    #             train[test_index] = pre
    #             test_pre[i, :] = pred
    #             cv_scores.append(log_loss(te_y, pre))
                
    #             f1_list=f1_score(te_y,np.argmax(pre,axis=1),average=None)
    #             f1=0.2*f1_list[0]+0.2*f1_list[1]+0.6*f1_list[2]
    #             f1_scores.append(f1)
    #             cv_rounds.append(model.best_iteration)
    #             test_pre_all[i, :] = np.argmax(pred, axis=1)

    #     print("%s now score is:" % clf_name, cv_scores)
    #     print("%s now f1-score is:" % clf_name, f1_scores)
    #     print("%s now round is:" % clf_name, cv_rounds)
    # test[:] = test_pre.mean(axis=0)
    # print("%s_score_list:" % clf_name, cv_scores)
    # print("%s_score_mean:" % clf_name, np.mean(cv_scores), np.mean(f1_scores))
    # print("%s_score_mean:" % clf_name, np.std(cv_scores))
    # return train, test, test_pre_all, np.mean(f1_scores)


def lgb(x_train, y_train, x_valid):
    lgb_train, lgb_test, sb, cv_scores = stacking(lightgbm, x_train, y_train, x_valid, "lgb", 3)
    return lgb_train, lgb_test, sb, cv_scores

path="/data4/mjx/gd/raw_data/"   #存放数据的地址
result_path="./"   #存放数据的地址
train_json=pd.read_json(path+"amap_traffic_annotations_train.json")
test_json=pd.read_json(path+"amap_traffic_annotations_test.json")

train_df=get_data(train_json[:],"/data4/mjx/gd/raw_data/amap_traffic_train_0712")
test_df=get_data(test_json[:],"/data4/mjx/gd/raw_data/amap_traffic_test_0712")

train_df.to_csv('feature/baseline_train.csv',index=False)
test_df.to_csv('feature/baseline_test.csv',index=False)
train_df = pd.read_csv('feature/baseline_train.csv')
test_df = pd.read_csv('feature/baseline_test.csv')
# # print(len(list(test_df["map_id"])))
# # print(set(list(np.arange(1,601)))-set(list(test_df["map_id"])))
train_df_self = pd.read_csv("feature/train_all_add_w.csv")
test_df_self = pd.read_csv("feature/test_all_add_w.csv")

train_df_time = pd.read_csv('feature/train_key_feature_time.csv')
test_df_time = pd.read_csv('feature/test_key_feature_time.csv')

train_df_ave_feature = pd.read_csv('feature/train_key_add_w.csv')
test_df_ave_feature = pd.read_csv('feature/test_key_add_w.csv')

# # train_key_detect_feature = pd.read_csv('feature/train_key_feature_detect.csv')
# # test_key_detect_feature = pd.read_csv('feature/test_key_feature_detect.csv')

# # train_feature_qyl = pd.read_csv('feature/train_Frtqyl.csv')
# # test_feature_qyl = pd.read_csv('feature/test_Frtqyl.csv')

# train_temp = pd.concat([train_df_self, train_df_time, train_df_ave_feature], axis=1)
# test_temp = pd.concat([test_df_self, test_df_time, test_df_ave_feature], axis=1)
# train_temp = train_temp[:].T.drop_duplicates().T
# test_temp = test_temp[:].T.drop_duplicates().T

# train_final = pd.merge(train_df, train_temp)
# test_final = pd.merge(test_df, test_temp)

# with open('7498.json') as f:
#         submit = json.load(f)
# submit_annos=submit['annotations']
# for i in range(len(submit_annos)):
#     submit_anno = submit_annos[i]
#     img_id = submit_anno['id']
#     status = submit_anno['status']
#     test_final.loc[test_final.map_id == int(img_id), 'label'] = status

# train_final.to_csv('feature/train_mjx.csv', index=False)
# test_final.to_csv('feature/test_mjx.csv',index=False)
# train_final = pd.read_csv('feature/train_mjx.csv')
# test_final = pd.read_csv('feature/test_mjx.csv')
# test_final["map_id"] = test_final["map_id"] + 1500
# train_w = pd.concat([train_final, test_final], axis=0)
# train_w.to_csv('feature/all.csv', index=False)

select_features=[
                "gap_mean","gap_std",
                 "hour_mean","minute_mean","dayofweek_mean",
                 "gap_time_today_mean","gap_time_today_std",
                 "P2","P3", "P1",
                 "key_hour", "key_minute", "key_day", "key_dayofweek",
                 "ave_P1", "ave_P3", "ave_P2"
                #  'box_0_min', 'box_1_min', 'box_2_max', 
                #  'box_3_max', 'car_num', 'box_center_gap_mean', 
                #  'box_center_gap_max', 'box_center_gap_min', 
                #  'box_center_gap_std', 'box_center_gap_range', 
                #  'last_distance', 'mask_area_mean', 
                #  'mask_area_max', 'mask_area_min', 
                #  'mask_area_std', 'mask_area_range', 
                #  'mask_area_all', 'mask_road'
                ]
# not_selected = ["label"]
# select_features_test = [i for i in train_final.columns.values if i not in not_selected]
train_w = pd.read_csv('feature/all.csv')
train_x=train_w[select_features].copy()
train_y=train_w["label"]
valid_x=test_final[select_features].copy()
folds = 5
seed = 2019
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
for train1, test1 in kf.split(train_x, train_y):
    print(len(train1), len(test1))
    print(train_y[test1])


# lgb_train, lgb_test, sb, m=lgb(train_x, train_y, valid_x)
# sub=test_final[["map_id"]].copy()
# sub["pred"]=np.argmax(lgb_test,axis=1)

# result_dic=dict(zip(sub["map_id"],sub["pred"]))

# # 保存
# import json
# with open(path+"amap_traffic_annotations_test.json","r") as f:
#     content=f.read()
# content=json.loads(content)
# for i in content["annotations"]:
#     i['status']=result_dic[int(i["id"])]
# print(m)
# with open(result_path+"sub_%s.json"%m,"w") as f:
#     f.write(json.dumps(content))


                
