import os
import copy
import pandas as pd
from gensim.models import Word2Vec
from random import shuffle
from keras.preprocessing.text import text_to_word_sequence
import numpy as np
import logging
import time
from tqdm import tqdm
import gc
"""
logging.basicConfig(level=logging.INFO)
used_cols = ['param_1','param_2','param_3','title', 'description']

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


all_samples['text'] = all_samples['param_1'].str.cat([all_samples.param_2,all_samples.param_3,all_samples.title,all_samples.description], sep=' ',na_rep='')
#all_samples.drop(used_cols, axis = 1, inplace=True)
del train_active,test_active,train,test
for i in used_cols:
    del all_samples[i]
gc.collect()
all_samples = all_samples['text'].values

all_samples = [text_to_word_sequence(text) for text in tqdm(all_samples)]
gc.collect()

model = Word2Vec(all_samples,size=100, window=5,workers=16,max_vocab_size=500000)
model.save('avito.w2v')
"""


def word2vec_type(x,type):
    wv_list=[]
    for w in x:
        if w in w2v.wv.vocab:
            wv=w2v[w]
            #print(wv)
            wv_list.append(wv)
    if wv_list:
        if type=="mean":
            return np.mean(np.array(wv_list),axis=0)
        if type=="sum":
            return np.sum(np.array(wv_list),axis=0)
        if type=="std":
            return np.std(np.array(wv_list),axis=0)
    else:
        return np.array([-1]*emb_size)

#获取平均词向量
#***************************************************
used_cols = ['item_id','param_1','param_2','param_3','title', 'description']
emb_size=100
train = pd.read_csv('../input/train.csv', usecols=used_cols).fillna("")
test = pd.read_csv('../input/test.csv', usecols=used_cols).fillna("")

target_samples = pd.concat([
    train,
    test,
]).reset_index(drop=True)


w2v = Word2Vec.load("avito.w2v")

for col in ['param_1','param_2','param_3','title', 'description']:
    train_array=np.empty((target_samples.shape[0],emb_size))
    r=0
    for row in target_samples[col]:
        train_array[r,:]=word2vec_type(text_to_word_sequence(row),"mean")
        r+=1

    embedding=pd.DataFrame(train_array)
    embedding.columns=["%s_embedding_%s"%(col,i) for i in range(emb_size)]
    embedding["item_id"]=target_samples["item_id"]
    embedding.to_csv("../embedding/%s_embedding.csv"%col,index=None)
