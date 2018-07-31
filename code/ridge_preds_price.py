#encoding=utf8
import time

notebookstart = time.time()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

print("Data:\n", os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

#wordbatch
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

NFOLDS = 5
SEED = 42
VALID = True


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if (seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


import string
stopwords = {x: 1 for x in stopwords.words('russian')}
def normalize_text(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return u' '.join(x for x in text if len(x) > 1 and x not in stopwords)

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))


print("\nData Load Stage")
text_cols = ['param_1','param_2','param_3','title','description']
print('loading train...')
train_a = pd.read_csv('../input/train_active.csv', usecols = text_cols+['item_id','price'])#[:100000]
print('loading test')
test_a = pd.read_csv('../input/test_active.csv',  usecols = text_cols+['item_id','price'])#[:100000]
print('concat dfs')

df_train = train_a.append(test_a)
df_train = df_train[pd.notnull(df_train['price'])].copy().reset_index(drop=True)
del train_a, test_a
gc.collect()


print('loading train...')
train = pd.read_csv('../input/train.csv', usecols = text_cols+['item_id','price'])
print('loading test')
test = pd.read_csv('../input/test.csv', usecols = text_cols+['item_id','price'])
print('concat dfs')
df_test = train.append(test).reset_index(drop=True)
del train, test
gc.collect()

for seed in range(10):
    df_train=df_train.sample(frac=0.2,random_state=2018)

y = df_train["price"].apply(np.log1p)

df=df_test[["item_id"]].copy()

for cols in ["title","description"]:
    df_train[cols] = df_train[cols].astype(str)
    df_train[cols] = df_train[cols].astype(str).fillna('missing') # FILL NA
    df_train[cols] = df_train[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently

    df_test[cols] = df_test[cols].astype(str)
    df_test[cols] = df_test[cols].astype(str).fillna('missing') # FILL NA
    df_test[cols] = df_test[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently

#####################################################title##########################################################
start_time=time.time()
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29,
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze = True
X_name_train = wb.fit_transform(df_train['title'])
print(X_name_train.shape)
X_name_test = wb.transform(df_test['title'])
print(X_name_test.shape)
del(wb)
gc.collect()

mask = np.where(X_name_train.getnnz(axis=0) > 3)[0]
X_name_train = X_name_train[:, mask]
print(X_name_train.shape)
X_name_test = X_name_test[:, mask]
print(X_name_test.shape)
print('[{}] Vectorize `title` completed.'.format(time.time() - start_time))


from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha': 30.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}

# Ridge oof method from Faron's kernel
# I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
# It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ridge.train(X_name_train, y)
df['ridge_preds_title'] = ridge.predict(X_name_test)

#####################################################description##########################################################
start_time=time.time()
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28,
                                                              "norm": "l2",
                                                              "tf": 1.0,
                                                              "idf": None}), procs=8)
wb.dictionary_freeze = True
X_description_train = wb.fit_transform(df_train['description'].fillna(''))
print(X_description_train.shape)
X_description_test = wb.transform(df_test['description'].fillna(''))
print(X_description_test.shape)
print('-')
del(wb)
gc.collect()

mask = np.where(X_description_train.getnnz(axis=0) > 8)[0]
X_description_train = X_description_train[:, mask]
print(X_description_train.shape)
X_description_test = X_description_test[:, mask]
print(X_description_test.shape)
print('[{}] Vectorize `description` completed.'.format(time.time() - start_time))


ridge_params = {'alpha': 30.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}

# Ridge oof method from Faron's kernel
# I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
# It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ridge.train(X_description_train, y)
df['ridge_preds_description'] = ridge.predict(X_description_test)


df[["item_id","ridge_preds_title","ridge_preds_description"]].to_csv("../input/ridge_preds_title_description.csv",index=None)
