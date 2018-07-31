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
df_train = pd.read_csv('../input/train.csv',parse_dates=["activation_date"])
df_test = pd.read_csv('../input/test.csv',parse_dates=["activation_date"])


df=df_train[["item_id"]].append(df_test[["item_id"]]).reset_index(drop=True).copy()

for cols in ["title","description"]:
    df_train[cols] = df_train[cols].astype(str)
    df_train[cols] = df_train[cols].astype(str).fillna('missing') # FILL NA
    df_train[cols] = df_train[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently

    df_test[cols] = df_test[cols].astype(str)
    df_test[cols] = df_test[cols].astype(str).fillna('missing') # FILL NA
    df_test[cols] = df_test[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently


ntrain=df_train.shape[0]
ntest=df_test.shape[0]

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = df_train.deal_probability.copy()

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
ridge_oof_train, ridge_oof_test = get_oof(ridge, X_name_train, y, X_name_test)

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds_title'] = ridge_preds

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


from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha': 30.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}

# Ridge oof method from Faron's kernel
# I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
# It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, X_description_train, y, X_description_test)

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))
print("Modeling Stage")
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds_description'] = ridge_preds


df[["item_id","ridge_preds_title","ridge_preds_description"]].to_csv("../input/ridge_preds_title_description.csv",index=None)
