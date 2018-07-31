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
valid=train[(train.activation_date>="2017-03-15")&(train.activation_date<="2017-03-21")].copy().reset_index(drop=True)
train=train[(train.activation_date>="2017-03-22")&(train.activation_date<="2017-03-28")].copy().reset_index(drop=True)

test=pd.read_csv("../input/test.csv")
test["deal_probability"]=-1
sub=test[["item_id"]].copy()

data_small=train.append(valid).append(test).reset_index(drop=True)

text_features=data_small[["item_id"]].copy()



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

#count features
count_features=[]
train,col=get_type_feature_all(train,data_small,["user_id"],"item_id","count","ori")
valid,col=get_type_feature_all(valid,data_small,["user_id"],"item_id","count","ori")
test,col=get_type_feature_all(test,data_small,["user_id"],"item_id","count","ori")
count_features.append(col)
train,col=get_type_feature_all(train,data_small,["user_id"],"activation_date","nunique","ori")
valid,col=get_type_feature_all(valid,data_small,["user_id"],"activation_date","nunique","ori")
test,col=get_type_feature_all(test,data_small,["user_id"],"activation_date","nunique","ori")
count_features.append(col)
train,col=get_type_feature_all(train,data_small,["category_name","image_top_1"],"item_id","count","ori")
valid,col=get_type_feature_all(valid,data_small,["category_name","image_top_1"],"item_id","count","ori")
test,col=get_type_feature_all(test,data_small,["category_name","image_top_1"],"item_id","count","ori")
count_features.append(col)


def create_tfidf_features(train,valid,test,data_small,col,n_comp):
    try:
        print("load tfidf..."+col)
        with open("../pickle/tfidf_%s_%s_train.pkl"%(col,n_comp),"rb") as f:
            df=pickle.load(f)
            for i in df.columns:
                train[i]=df[i]
        with open("../pickle/tfidf_%s_%s_valid.pkl"%(col,n_comp),"rb") as f:
            df=pickle.load(f)
            for i in df.columns:
                valid[i]=df[i]
        with open("../pickle/tfidf_%s_%s_test.pkl"%(col,n_comp),"rb") as f:
            df=pickle.load(f)
            for i in df.columns:
                test[i]=df[i]
    except:
        print("get tfidf..."+col)
        tfidf_vec = TfidfVectorizer(ngram_range=(0, 1))
        full_tfidf=tfidf_vec.fit_transform(data_small[col].values.tolist())
        svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
        tfidf_svd=pd.DataFrame(svd_obj.fit_transform(full_tfidf))
        columns=[col+'_tfidf_svd_'+str(i+1) for i in range(n_comp)]
        tfidf_svd.columns=columns
        tfidf_svd["item_id"]=data_small["item_id"]

        train=train.merge(tfidf_svd,on="item_id",how="left")
        valid=valid.merge(tfidf_svd,on="item_id",how="left")
        test=test.merge(tfidf_svd,on="item_id",how="left")
        with open("../pickle/tfidf_%s_%s_train.pkl"%(col,n_comp),"wb") as f:
            pickle.dump(train[columns],f)
        with open("../pickle/tfidf_%s_%s_valid.pkl"%(col,n_comp),"wb") as f:
            pickle.dump(valid[columns],f)
        with open("../pickle/tfidf_%s_%s_test.pkl"%(col,n_comp),"wb") as f:
            pickle.dump(test[columns],f)
    return train,valid,test


def create_category_features(train,valid,test,data_small,col):
    try:
        print("load category..."+col)
        with open("../pickle/%s_train.pkl"%(col),"rb") as f:
            df=pickle.load(f)
            for i in df.columns:
                train[i]=df[i]
        with open("../pickle/%s_valid.pkl"%(col),"rb") as f:
            df=pickle.load(f)
            for i in df.columns:
                valid[i]=df[i]
        with open("../pickle/%s_test.pkl"%(col),"rb") as f:
            df=pickle.load(f)
            for i in df.columns:
                test[i]=df[i]
    except:
        print("get category..."+col)
        lbl = LabelEncoder()
        cate=lbl.fit_transform(data_small[col])
        cate=pd.DataFrame(cate)
        cate.columns=[col]
        cate["item_id"]=data_small["item_id"]
        del train[col],valid[col],test[col]
        train=train.merge(cate,on="item_id",how="left")
        valid=valid.merge(cate,on="item_id",how="left")
        test=test.merge(cate,on="item_id",how="left")
        with open("../pickle/%s_train.pkl"%(col),"wb") as f:
            pickle.dump(train[[col]],f)
        with open("../pickle/%s_valid.pkl"%(col),"wb") as f:
            pickle.dump(valid[[col]],f)
        with open("../pickle/%s_test.pkl"%(col),"wb") as f:
            pickle.dump(test[[col]],f)

    return train,valid,test

for col in ["title","description"]:
    data_small[col]=data_small[col].astype(str)
    train,valid,test=create_tfidf_features(train,valid,test,data_small,col,5)

cate_features=["region", "city", "parent_category_name", "category_name", "user_type", "param_1","param_2","param_3"]
for col in cate_features:
    data_small[col]=data_small[col].astype(str)
    train,valid,test=create_category_features(train,valid,test,data_small,col)



#process features
"""
data_small['num_words_title'] = data_small['title'].apply(lambda comment: len(comment.split()))
data_small['num_unique_words_title'] = data_small['title'].apply(lambda comment: len(set(w for w in comment.split())))

data_small['num_words_description'] = data_small['description'].apply(lambda comment: len(comment.split()))
data_small['num_unique_words_description'] = data_small['description'].apply(lambda comment: len(set(w for w in comment.split())))

data_small['words_vs_unique_title'] = data_small['num_unique_words_title'] / data_small['num_words_title'] * 100
data_small['words_vs_unique_description'] = data_small['num_unique_words_description'] / data_small['num_words_description'] * 100

process_features=['num_words_title','num_unique_words_title','num_words_description','num_unique_words_description','words_vs_unique_title','words_vs_unique_description']
train=train.merge(data_small[['item_id']+process_features],on='item_id',how="left")
valid=valid.merge(data_small[['item_id']+process_features],on='item_id',how="left")
test=test.merge(data_small[['item_id']+process_features],on='item_id',how="left")
"""
process_features=[]

#agg features
"""
agg=pd.read_csv("../input/aggregated_features.csv")
agg_features=[i for i in agg.columns if i!="user_id"]
train=train.merge(agg,on="user_id",how="left")
valid=valid.merge(agg,on="user_id",how="left")
test=test.merge(agg,on="user_id",how="left")
"""
agg_features=[]

#tfidf_features=[i for i in train.columns if "_tfidf_svd_" in i]
tfidf_features=[]
num_features=["price","image_top_1","item_seq_number"]
features=num_features+tfidf_features+cate_features+count_features+process_features+agg_features
target="deal_probability"

train_y=train[target].values
valid_y=valid[target].values

train=train[features].values
valid=valid[features].values
test=test[features].values

#################################
train_matrix = lgb.Dataset(train, label=train_y)
valid_matrix = lgb.Dataset(valid, label=valid_y)

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

num_round = 20000
early_stopping_rounds = 100
model = lgb.train(params, train_matrix, num_round, valid_sets=valid_matrix,
                  early_stopping_rounds=early_stopping_rounds
                  )
print("\n".join(("%s: %.2f" % x) for x in sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1],reverse=True)))
pre_valid=model.predict(valid)
score=mean_squared_error(valid_y,pre_valid)**0.5
print(score)
with open("score.txt","a") as f:
    f.write(str(score)+"\n")

sub[target]=np.clip(model.predict(test), 0, 1)
sub.to_csv("../sub/sub_%s.csv"%str(score),index=None)
