#encoding=utf8
import pandas as pd
import numpy as np
import lightgbm as lgb

import gc
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
test["deal_probability"]=-1

sub=test[["item_id"]].copy()
train_length=train.shape[0]

data_small=train.append(test).reset_index(drop=True)


def get_type_feature_all(sample,train_df,key,on,type_c,mark):
    filename = "_".join([mark+"_%s_features"%type_c, "_".join(key), on, str(len(sample))]) + ".pkl"
    try:
        with open("../pickle/" + filename, "rb") as fp:
            print("load {} {} feature from pickle file: key: {}, on: {}...".format(mark,type_c,"_".join(key), on))
            col = pickle.load(fp)
        for c in col.columns:
            sample[c] = col[c]
        gc.collect()
    except:
        print('get {} {} feature, key: {}, on: {}'.format(mark,type_c,"_".join(key), on))
        if type_c=="count":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].count()).reset_index()
        if type_c=="mean":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].mean()).reset_index()
        if type_c=="nunique":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].nunique()).reset_index()
        if type_c=="max":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].max()).reset_index()
        if type_c=="min":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].min()).reset_index()
        if type_c=="sum":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].sum()).reset_index()
        if type_c=="std":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].std()).reset_index()
        if type_c=="median":
            tmp = pd.DataFrame(train_df[key+[on]].groupby(key)[on].median()).reset_index()
        tmp.columns = key+[mark+"_"+"_".join(key) + '_%s_'%type_c + on]
        tmp[mark+"_"+"_".join(key) + '_%s_'%type_c + on] = tmp[mark+"_"+"_".join(key) + '_%s_'%type_c + on].astype('float32')
        sample = sample.merge(tmp, on=key, how='left')
        with open("../pickle/" + filename, "wb") as fp:
            col = sample[[mark+"_"+"_".join(key) + '_%s_'%type_c + on]]
            pickle.dump(col, fp)
        del tmp
    del col,train_df
    gc.collect()
    return sample,mark+"_"+"_".join(key) + '_%s_'%type_c + on


cate_features=["region", "city", "parent_category_name", "category_name", "user_type", "param_1","param_2","param_3"]
for col in cate_features:
    data_small[col]=data_small[col].astype(str)
    lbl = LabelEncoder()
    data_small[col] = lbl.fit_transform(data_small[col])


num_features=["price","image_top_1","item_seq_number"]
features=num_features+cate_features
target="deal_probability"


train=data_small[:train_length]
test=data_small[train_length:]

train_y=train[target].values

train=train[features].values
test=test[features].values

#################################


def stacking(clf,train_x,train_y,test_x,clf_name,class_num=1):
    train=np.zeros((train_x.shape[0],class_num))
    test=np.zeros((test_x.shape[0],class_num))
    test_pre=np.zeros((folds,test_x.shape[0],class_num))
    cv_scores=[]
    cv_rounds=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]

        train_matrix = clf.Dataset(tr_x, label=tr_y)
        test_matrix = clf.Dataset(te_x, label=te_y)

        params = {
                  'boosting_type': 'gbdt',
                  'objective': 'regression_l2',
                  'metric': 'rmse',
                  'min_child_weight': 1.5,
                  'num_leaves': 2**5,
                  'lambda_l2': 10,
                  'subsample': 0.7,
                  'colsample_bytree': 0.5,
                  'colsample_bylevel': 0.5,
                  'learning_rate': 0.02,
                  'seed': 2017,
                  'nthread': 16,
                  'silent': True,
                  }


        num_round = 5000
        early_stopping_rounds = 100
        if test_matrix:
            model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                              early_stopping_rounds=early_stopping_rounds
                              )

            print("\n".join(("%s: %.2f" % x) for x in
                            sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1],
                                   reverse=True)))

            pre= model.predict(te_x,num_iteration=model.best_iteration).reshape((te_x.shape[0],1))
            train[test_index]=pre
            test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape((test_x.shape[0],1))
            cv_scores.append(mean_squared_error(te_y, pre)**0.5)
            cv_rounds.append(model.best_iteration)

        print("%s now score is:"%clf_name,cv_scores)
        print("%s now round is:"%clf_name,cv_rounds)
    test[:]=test_pre.mean(axis=0)
    print("%s_score_list:"%clf_name,cv_scores)
    print("%s_score_mean:"%clf_name,np.mean(cv_scores))
    with open("score_cv.txt", "a") as f:
        f.write("%s now score is:" % clf_name + str(cv_scores) + "\n")
        f.write(str(cv_rounds)+"----------")
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1),test.reshape(-1),np.mean(cv_scores)


def lgb(x_train, y_train, x_valid):
    xgb_train, xgb_test,cv_scores = stacking(lightgbm, x_train, y_train, x_valid,"lgb")
    return xgb_train, xgb_test,cv_scores

import lightgbm
from sklearn.cross_validation import KFold
folds = 5
seed = 2018

kf = KFold(train.shape[0], n_folds=folds, shuffle=True, random_state=seed)
lgb_train, lgb_test,m=lgb(train, train_y, test)


sub[target]=np.clip(lgb_test, 0, 1)
sub.to_csv("../sub/sub_%s.csv"%str(m),index=None)
