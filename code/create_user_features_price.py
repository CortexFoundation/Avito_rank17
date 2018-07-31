import pandas as pd
import numpy as np
import gc

used_cols = ['user_id','category_name','price']

train_active = pd.read_csv('../input/train_active.csv', usecols=used_cols)
test_active = pd.read_csv('../input/test_active.csv', usecols=used_cols)

all_samples = pd.concat([train_active,test_active]).dropna().reset_index(drop=True)
all_samples["price"]=all_samples["price"].apply(np.log1p)


del train_active
del test_active
gc.collect()
gp=pd.DataFrame(all_samples.groupby("user_id").price.mean()).reset_index()
gp.columns=["user_id","user_price_mean"]
gp.to_csv('../input/user_price_mean.csv', index=False)

gp=pd.DataFrame(all_samples.groupby(["user_id","category_name"]).price.mean()).reset_index()
gp.columns=["user_id","category_name","user_cate_price_mean"]
gp.to_csv('../input/user_cate_price_mean.csv', index=False)
