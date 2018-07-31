#encoding=utf8
import pandas as pd
import numpy as np
import time
import gc
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

text_cols = ['param_1','param_2','param_3','title','description']
print('loading train...')
train_a = pd.read_csv('../input/train_active.csv', usecols = text_cols+['item_id','category_name'])
print('loading test')
test_a = pd.read_csv('../input/test_active.csv',  usecols = text_cols+['item_id','category_name'])
print('concat dfs')
df_train = train_a.append(test_a)
df_train = df_train[pd.notnull(df_train['category_name'])].copy().reset_index(drop=True)
del train_a, test_a
gc.collect()

print('loading train...')
train = pd.read_csv('../input/train.csv', usecols = text_cols+['item_id','category_name'])
print('loading test')
test = pd.read_csv('../input/test.csv', usecols = text_cols+['item_id','category_name'])
print('concat dfs')
df_test = train.append(test).reset_index(drop=True)
del train, test
gc.collect()

train_length=df_train.shape[0]
df=df_train.append(df_test).reset_index(drop=True)
test_id=df[train_length:][["item_id"]].copy()

test_pred=pd.read_csv("../input/label_cate_features.csv")
del test_pred["item_id"]
test_pred["item_id"]=list(test_id["item_id"])
test_pred.to_csv('../input/label_cate_features.csv',index=None)
