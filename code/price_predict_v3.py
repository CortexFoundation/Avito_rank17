#encoding=utf8
import pandas as pd
import numpy as np
import time
import gc
text_cols = ['parent_category_name', 'category_name','param_1','param_2','param_3','title','description']
print('loading train...')
train_a = pd.read_csv('../input/train_active.csv', usecols = text_cols+['item_id','price'])
print('loading test')
test_a = pd.read_csv('../input/test_active.csv',  usecols = text_cols+['item_id','price'])
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

train_length=df_train.shape[0]
df=df_train.append(df_test).reset_index(drop=True)

print('cleaning text')

for col in text_cols:
    df[col] = df[col].fillna('nan').astype(str)
print('concat text')
df['text'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)
#df['text'] = df["description"]
df.drop(text_cols,axis = 1, inplace = True)

from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences

max_features = 100000 # max amount of words considered
max_len = 100 #maximum length of text
dim = 100 #dimension of embedding


print('tokenizing...',end='')
tic = time.time()
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['text'].values))
toc = time.time()
print('done. {}'.format(toc-tic))

col = 'text'
print("   Transforming {} to seq...".format(col))
tic = time.time()
df[col] = tokenizer.texts_to_sequences(df[col])
toc = time.time()
print('done. {}'.format(toc-tic))

print('padding X_train')
tic = time.time()
X_train = pad_sequences(df[:train_length][col], maxlen=max_len)
toc = time.time()
print('done. {}'.format(toc-tic))
#train_id=df[:train_length][["item_id"]].copy()

print('padding X_nan')
tic = time.time()
X_nan = pad_sequences(df[train_length:][col], maxlen=max_len)
toc = time.time()
print('done. {}'.format(toc-tic))
test_id=df[train_length:][["item_id"]].copy()

df.drop(['text'], axis = 1, inplace=True)

y = df[:train_length]["price"].apply(np.log1p).values

gc.collect()

import numpy as np
from keras.layers import Input,PReLU,BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.layers import Concatenate, Flatten, Bidirectional
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from attention import Attention


from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy



def all_pool(tensor):
    avg_tensor = GlobalAveragePooling1D()(tensor)
    max_tensor = GlobalMaxPooling1D()(tensor)
    res_tensor = Concatenate()([avg_tensor, max_tensor])
    return res_tensor

def build_model():
    inp = Input(shape=(max_len,))

    embedding = Embedding(max_features + 1, dim)(inp)
    x = Bidirectional(CuDNNGRU(64,return_sequences=True))(embedding)
    x = Attention()(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation = 'relu')(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer=Adam(lr=0.002), loss="mean_squared_error")
    return model

model = build_model()
model.summary()

early_stop = EarlyStopping(patience=2)
check_point = ModelCheckpoint('price_model_v3.hdf5', monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)

history = model.fit(X_train, y, batch_size = 10240, epochs = 30,verbose = 1, validation_split=0.1,callbacks=[early_stop,check_point])

model.load_weights('price_model_v3.hdf5')
preds = model.predict(X_nan,verbose=1)

test_id["price_pred_all"]=preds.reshape(-1)
test_id.to_csv('../input/label_price_features_all.csv',index=None)
