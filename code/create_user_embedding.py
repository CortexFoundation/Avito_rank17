import pandas as pd
import gc

used_cols = ['item_id',"param_1", "param_2", "param_3","parent_category_name", "category_name",
             'user_id', "region", "city", "user_type"]

train = pd.read_csv('../input/train.csv', usecols=used_cols)
train_active = pd.read_csv('../input/train_active.csv', usecols=used_cols)
test = pd.read_csv('../input/test.csv', usecols=used_cols)
test_active = pd.read_csv('../input/test_active.csv', usecols=used_cols)

all_samples = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)

target_samples = pd.concat([
    train,
    test,
]).reset_index(drop=True)

del train_active
del test_active
gc.collect()

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
        df[new_column]=(df[column]==value).astype("int8")
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
    return sample

def get_cat_feature_gp(df,key,column,value):
    print("get cat feature from pickle_sample_small file: key: {}, on: {}...".format("_".join(key),column + "_" + str(value)))
    new_column=column + "_" + str(value)
    df[new_column]=(df[column]==value).astype("int8")
    gp=pd.DataFrame(df.groupby(key)[new_column].mean()).reset_index()
    gp.columns=key+["cat_features_"+"_".join(key)+"_"+new_column]
    return gp

for key in ["user_id"]:
    data=target_samples[[key]].drop_duplicates().copy()
    for j in list(all_samples["parent_category_name"].value_counts().index):
        gp=get_cat_feature_gp(all_samples,[key],"parent_category_name",j)
        data=data.merge(gp,on=key,how="left").fillna(0)
    data.to_csv("../embedding/user_parent_cate_embedding.csv",index=None)

    data = target_samples[[key]].drop_duplicates().copy()
    for j in list(all_samples["category_name"].value_counts().index):
        gp=get_cat_feature_gp(all_samples,[key],"category_name",j)
        data=data.merge(gp,on=key,how="left").fillna(0)
    data.to_csv("../embedding/user_cate_embedding.csv",index=None)
