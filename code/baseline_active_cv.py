#encoding=utf8
import pandas as pd
import numpy as np
import lightgbm as lgb
import time

import gc
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
test["deal_probability"]=-1


####
np.random.seed(2018)
class TargetEncoder:
    # Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def __repr__(self):
        return 'TargetEncoder'

    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.keep_original = keep_original

    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self, train, test, target):
        for col in self.cols:
            if self.keep_original:
                train[col + '_te'], test[col + '_te'] = self.encode_column(train[col], test[col], target)
            else:
                train[col], test[col] = self.encode_column(train[col], test[col], target)
        return train, test

    def encode_column(self, trn_series, tst_series, target):
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)

#for i in [""]
f_cats = ["region","city","parent_category_name","category_name","user_type","image_top_1"]
target_encode = TargetEncoder(min_samples_leaf=100, smoothing=10, noise_level=0.01,
                              keep_original=True, cols=f_cats)
train, test = target_encode.encode(train, test, train["deal_probability"])
print(train.head())
f_cats_encode=["%s_te"%i for i in f_cats]
####


def get_type_feature(train_df, key, on,type_c):
    filename = "_".join(["%s_features"%type_c, "_".join(key), on, str(len(train_df))]) + ".pkl"
    try:
        with open("../pickle/" + filename, "rb") as fp:
            print("load {} feature from pickle file: key: {}, on: {}...".format(type_c,"_".join(key), on))
            col = pickle.load(fp)
        for c in col.columns:
            train_df[c] = col[c]
        gc.collect()
    except:
        print('get {} feature, key: {}, on: {}'.format(type_c,"_".join(key), on))
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
        tmp.columns = key+["_".join(key) + '_%s_'%type_c + on]
        tmp["_".join(key) + '_%s_'%type_c + on] = tmp["_".join(key) + '_%s_'%type_c + on].astype('float32')
        train_df = train_df.merge(tmp, on=key, how='left', copy=False)
        with open("../pickle/" + filename, "wb") as fp:
            col = train_df[["_".join(key) + '_%s_'%type_c + on]]
            pickle.dump(col, fp)
        del tmp
    del col
    gc.collect()
    return train_df,"_".join(key) + '_%s_'%type_c + on

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


data_small=train.append(test).reset_index(drop=True)
ts=pd.to_datetime(data_small["activation_date"])
data_small["activation_weekday"]=ts.dt.weekday
data_small["activation_dayofyear"]=ts.dt.dayofyear
data_small["price"]=data_small["price"].apply(np.log1p)
#del data_small["image_top_1"]


ui_features=[]
user_features=pd.read_csv("../input/user_features.csv")
data_small=data_small.merge(user_features,on="user_id",how="left")
ui_features+=[i for i in user_features.columns if i!="user_id"]

user_features=pd.read_csv("../embedding/user_cate_embedding.csv").drop_duplicates().copy()
data_small=data_small.merge(user_features,on="user_id",how="left")
cate=[i for i in user_features.columns if i!="user_id"]
ui_features+=cate


item_features=pd.read_csv("../input/label_price_features.csv")[["item_id","price_pred"]]
data_small=data_small.merge(item_features,on="item_id",how="left")
ui_features+=["price_pred"]
item_features=pd.read_csv("../input/label_price_features_onlydescription.csv")
data_small=data_small.merge(item_features,on="item_id",how="left")
ui_features+=[i for i in item_features.columns if i!="item_id"]
item_features=pd.read_csv("../input/label_price_features_all.csv")
data_small=data_small.merge(item_features,on="item_id",how="left")
ui_features+=[i for i in item_features.columns if i!="item_id"]
item_features=pd.read_csv("../input/ridge_preds.csv")
data_small=data_small.merge(item_features,on="item_id",how="left")
ui_features+=[i for i in item_features.columns if i!="item_id"]
item_features=pd.read_csv("../input/ridge_preds_title_description.csv")
data_small=data_small.merge(item_features,on="item_id",how="left")
ui_features+=[i for i in item_features.columns if i!="item_id"]
item_features=pd.read_csv("../input/label_price_features_globalaverage.csv")
item_features.columns=["item_id","ga"]
data_small=data_small.merge(item_features,on="item_id",how="left")
ui_features+=[i for i in item_features.columns if i!="item_id"]

"""
user_features=pd.read_csv("../input/user_price_mean.csv")
data_small=data_small.merge(user_features,on="user_id",how="left")
ui_features+=["user_price_mean"]

user_features=pd.read_csv("../input/user_cate_price_mean.csv")
data_small=data_small.merge(user_features,on=["user_id","category_name"],how="left")
ui_features+=["user_cate_price_mean"]
"""
#item_features=pd.read_csv("../input/label_price_features.csv")
#data_small=data_small.merge(item_features,on="item_id",how="left")
#ui_features+=[i for i in item_features.columns if i!="item_id"]


cate_features=["region", "city", "parent_category_name", "category_name", "user_type", "param_1","param_2","param_3"]
for col in cate_features:
    data_small[col]=data_small[col].astype(str)
    lbl = LabelEncoder()
    data_small[col] = lbl.fit_transform(data_small[col])

count_features=[]
for col in cate_features:
    for tar in ['price','image_top_1','item_seq_number','price_pred','price_pred_all','ridge_preds']:
        data_small,fe=get_type_feature(data_small,[col],tar,"mean")
        count_features.append(fe)

for col in ["parent_category_name","category_name","param_1","param_2","param_3","activation_date"]:
    data_small,fe=get_type_feature(data_small,["user_id"],col,"nunique")
    count_features.append(fe)

data_small, fe = get_type_feature(data_small, ["user_id","activation_date"], "item_id", "count")
count_features.append(fe)

data_small, fe = get_type_feature(data_small, ["image_top_1"], "item_id", "nunique")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["image_top_1"], "user_id", "nunique")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["image_top_1"], "category_name", "nunique")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["image_top_1"], "param_1", "nunique")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["image_top_1"], "item_seq_number", "nunique")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["image_top_1"], "price_pred", "mean")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["image_top_1"], "price_pred", "std")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["image_top_1"], "item_seq_number", "mean")
count_features.append(fe)

data_small, fe = get_type_feature(data_small, ["user_id"], "ridge_preds", "mean")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["user_id","category_name"], "ridge_preds", "mean")
count_features.append(fe)
data_small, fe = get_type_feature(data_small, ["user_id","image_top_1"], "ridge_preds", "mean")
count_features.append(fe)

data_small, fe = get_type_feature(data_small, ["user_id","category_name"], "ridge_preds", "sum")
count_features.append(fe)


def get_cat_feature(sample,train_df,key,column,value):
    new_column=column + "_" + str(value)
    sample_filename = "_".join(["cat_features", "_".join(key), column,str(value), str(len(sample))]) + ".pkl"
    try:
        with open("../pickle/" + sample_filename, "rb") as fp:
            print("load cat feature from pickle_sample_small file: key: {}, on: {}...".format("_".join(key), column+"_"+str(value)))
            col = pickle.load(fp)
        for c in col.columns:
            sample[c] = col[c]
    except:
        print("get cat feature from pickle_sample_small file: key: {}, on: {}...".format("_".join(key), column + "_" + str(value)))
        df=train_df.copy()
        df[new_column]=df[column].apply(lambda x:1 if x==value else 0)
        gp=pd.DataFrame(df.groupby(key)[new_column].mean()).reset_index()
        gp.columns=key+["cat_features_"+"_".join(key)+"_"+new_column]
        sample=sample.merge(gp,on=key,how="left").fillna(0)
        with open("../pickle/" + sample_filename, "wb") as fp:
            col = sample[["cat_features_"+"_".join(key)+"_"+new_column]]
            pickle.dump(col, fp)
        del df
        del gp
        gc.collect()
    del train_df
    gc.collect()
    return sample,"cat_features_"+"_".join(key)+"_"+new_column

"""
for col in ["image_top_1"]:
    for i in list(data_small[col].value_counts().index[:10]):
        data_small,fe = get_cat_feature(data_small, data_small, ['user_id'], col, i)
        count_features.append(fe)
"""

tfidf_features=[]
n_comp=5
for col in ["title","description"]:
    try:
        with open("../pickle/tfidf_%s.pkl"%col,"rb") as f:
            tfidf_svd=pickle.load(f)
    except:
        data_small[col]=data_small[col].astype(str)
        tfidf_vec = TfidfVectorizer(ngram_range=(1, 2))
        full_tfidf = tfidf_vec.fit_transform(data_small[col].values.tolist())
        svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
        tfidf_svd = pd.DataFrame(svd_obj.fit_transform(full_tfidf))
        columns = [col + '_tfidf_svd_' + str(i + 1) for i in range(n_comp)]
        tfidf_svd.columns = columns
        with open("../pickle/tfidf_%s.pkl"%col,"wb") as f:
            pickle.dump(tfidf_svd,f)
    for c in tfidf_svd.columns:
        data_small[c]=tfidf_svd[c]
        tfidf_features.append(c)


num_features=["price","image_top_1","item_seq_number","activation_weekday"]+ui_features
features=num_features+cate_features+tfidf_features+count_features+f_cats_encode+shift_fe
target="deal_probability"



train=data_small[(data_small.activation_date>="2017-03-15")&(data_small.activation_date<="2017-03-28")].copy().reset_index(drop=True)
test=data_small[(data_small.activation_date>="2017-04-12")].copy().reset_index(drop=True)
sub=test[["item_id"]].copy()

y_train=train[target].values

x_train=train[features].values
x_test=test[features].values

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
            'num_leaves': 2 ** 5,
            'lambda_l2': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'colsample_bylevel': 0.5,
            'learning_rate': 0.1,
            'seed': 2018,
            'nthread': 16,
            'silent': True,
        }


        num_round = 16000
        early_stopping_rounds = 100
        if test_matrix:
            model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                              early_stopping_rounds=early_stopping_rounds
                              )

            print("\n".join(("%s: %.2f" % x) for x in
                            sorted(zip(predictors, model.feature_importance("gain")), key=lambda x: x[1],
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

predictors=features
kf = KFold(x_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)
lgb_train, lgb_test,m=lgb(x_train, y_train, x_test)

sub[target]=np.clip(lgb_test, 0, 1)
sub.to_csv("../sub/sub_cv_%s.csv"%str(m),index=None)


