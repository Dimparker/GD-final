from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import lightgbm
import lightgbm as lgb
import matplotlib.image as mpimg
from sklearn.metrics import f1_score, log_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, KFold
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

def gs(clf, train_x, train_y, test_x, clf_name, class_num=3):
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

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'min_child_weight': 1.5,
        'num_leaves': 2 ** 3 -1,
        'lambda_l2': 10,
        'bagging_freq': 4,
        'learning_rate': 0.05,
        'seed': 2019,
        'nthread': 28,
        'num_class': class_num,
        'verbose': -1,
    }
    params_test={
    'learning_rate': np.arange(0.03,0.1,0.01),
    }
    # train_matrix = clf.Dataset(train_x, label=train_y)
    # model = clf.train(params, train_matrix, 4000, verbose_eval=50)
    lf = lgb.LGBMClassifier(
                       num_leaves=7,
                       max_depth=3,
                       learning_rate=0.1,
                       n_estimators=4000,
                       objective='multiclass',
                       random_state=2020,
                       reg_lambda=10,
                    #    class_weight=[0.2,0.2,0.6], 
                    #    bagging_fraction = 0.8,
                    #    feature_fraction = 0.8，
                       min_child_weight=1.5, 
                       silent=True,
                       n_jobs=-1,
                       )         
   
    # score_way = make_scorer(my_scoring)
    gsearch = GridSearchCV(estimator=lf, param_grid=params_test,scoring='f1_weighted', cv=5, verbose=1)    
    gsearch.fit(train_x, train_y)
    print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)
    # cv_scores = []
    # f1_scores = []
    # cv_rounds = []

    # for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
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
    #             'metric': 'multi_logloss',
    #             'min_child_weight': 1.5,
    #             'num_leaves': 2 ** 3 -1,
    #             'lambda_l2': 10,
    #             # 'feature_fraction': 0.9,
    #             # 'bagging_fraction': 0.9,
    #             'bagging_freq': 4,
    #             'learning_rate': 0.05,
    #             'seed': 2019,
    #             'nthread': 28,
    #             'num_class': class_num,
    #             'verbose': -1,
    #         }

    #         num_round = 4000
    #         early_stopping_rounds = 100
    #         if test_matrix:
    #             model = clf.train(params, train_matrix, num_round, valid_sets=test_matrix, verbose_eval=50,
    #                               early_stopping_rounds=early_stopping_rounds
    #                               )
    #             print("feature importance:\n", "\n".join(("%s: %.2f" % x) for x in
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

    # test[:] = test_pre.mean(axis=0)
    # return train, test, test_pre_all, np.mean(f1_scores)

path="/data4/mjx/gd/raw_data/"   #存放数据的地址
result_path="./"   #存放数据的地址

train_df = pd.read_csv('feature/baseline_train.csv')
test_df = pd.read_csv('feature/baseline_test.csv')
train_df_self = pd.read_csv("feature/train_add.csv")
test_df_self = pd.read_csv("feature/test_add.csv")

train_final = pd.merge(train_df, train_df_self)
test_final = pd.merge(test_df, test_df_self)

select_features=[
                "gap_mean","gap_std",
                 "hour_mean","minute_mean","dayofweek_mean",
                 "gap_time_today_mean","gap_time_today_std",
                 "P1", "P2", "P3"
                ]

train_x=train_final[select_features].copy()
train_y=train_final["label"]
valid_x=test_final[select_features].copy()
gs(lightgbm, train_x, train_y, valid_x, "lgb", 3)
# lgb_train, lgb_test, sb, m=lgb(train_x, train_y, valid_x)
# sub=test_final[["map_id"]].copy()
# sub["pred"]=np.argmax(lgb_test,axis=1)

# result_dic=dict(zip(sub["map_id"],sub["pred"]))
# # print(result_dic)
# #保存
# import json
# with open(path+"amap_traffic_annotations_test.json","r") as f:
#     content=f.read()
# content=json.loads(content)
# for i in content["annotations"]:
#     i['status']=result_dic[int(i["id"])]
# with open(result_path+"sub_%s.json"%m,"w") as f:
#     f.write(json.dumps(content))

