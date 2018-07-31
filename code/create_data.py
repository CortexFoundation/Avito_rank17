#encoding=utf8
def get_data():
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
    """
    user_features=pd.read_csv("../embedding/user_parent_cate_embedding.csv").drop_duplicates().copy()
    data_small=data_small.merge(user_features,on="user_id",how="left")
    parent_cate=[i for i in user_features.columns if i!="user_id"]
    ui_features+=parent_cate
    """
    user_features=pd.read_csv("../embedding/user_cate_embedding.csv").drop_duplicates().copy()
    data_small=data_small.merge(user_features,on="user_id",how="left")
    cate=[i for i in user_features.columns if i!="user_id"]
    ui_features+=cate

    #for _ in cate:
    #    data_small, fe = get_type_feature(data_small, ["image_top_1"], _, "mean")
    #    ui_features.append(fe)


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

    item_features=pd.read_pickle("../input/ImageInfo")
    print(item_features)
    data_small=data_small.merge(item_features,on="image",how="left")
    ui_features+=[i for i in item_features.columns if i!="image"]

    item_features=pd.read_pickle("../input/image_size")
    print(item_features)
    data_small=data_small.merge(item_features,on="image",how="left")
    ui_features+=[i for i in item_features.columns if i!="image"]

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
    features=num_features+cate_features+tfidf_features+count_features+f_cats_encode
    target="deal_probability"

    #data_small[features+["item_id"]].to_csv("all_features_plantsgo.csv",index=None)

    data_small=data_small.fillna(-1)
    train=data_small[(data_small.activation_date>="2017-03-15")&(data_small.activation_date<="2017-03-28")].copy().reset_index(drop=True)
    test=data_small[(data_small.activation_date>="2017-04-12")].copy().reset_index(drop=True)
    y_train=train[target].values

    x_train=train[features].values
    x_test=test[features].values

    train=train[["item_id"]].copy()
    test=test[["item_id"]].copy()

    return x_train,x_test,y_train,train,test


if __name__=="__main__":
    get_data()


