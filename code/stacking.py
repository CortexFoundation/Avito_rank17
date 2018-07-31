#encoding=utf8
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from scipy import sparse
import xgboost
import lightgbm

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss,mean_absolute_error,mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB

#####################################################回归##################################################

###########################################################################################################

def stacking_reg(clf,train_x,train_y,test_x,clf_name):
    train=np.zeros((train_x.shape[0],1))
    test=np.zeros((test_x.shape[0],1))
    test_pre=np.empty((folds,test_x.shape[0],1))
    cv_scores=[]
    for i,(train_index,test_index) in enumerate(kf):
        tr_x=train_x[train_index]
        tr_y=train_y[train_index]
        te_x=train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ["rf","ada","gb","et","lr","lsvc","knn"]:
            clf.fit(tr_x,tr_y)
            #print(clf.coef_)
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_squared_error(te_y, pre)**0.5)
        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=te_y, missing=-1)
            params = {'booster': 'gbtree',
                      'eval_metric': 'rmse',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.1,
                      'tree_method': 'exact',
                      'seed': 2017,
                      'nthread': 16
                      }
            num_round = 35000
            early_stopping_rounds = 100
            watchlist = [(train_matrix, 'train'),
                         (test_matrix, 'eval')
                         ]
            if test_matrix:
                model = clf.train(params, train_matrix, num_boost_round=num_round,evals=watchlist,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(test_matrix,ntree_limit=model.best_ntree_limit).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(z, ntree_limit=model.best_ntree_limit).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre)**0.5)

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y)
            #z = clf.Dataset(test_x, label=te_y)
            #z=test_x
            params = {
                'num_leaves': 2 ** 5 - 1,
                'objective': 'regression_l2',
                'max_depth': 8,
                'min_data_in_leaf': 50,
                'learning_rate': 0.1,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.75,
                'bagging_freq': 1,
                'metric': 'rmse',
                'num_threads': 16,
                'seed': 2018
            }
            num_round = 35000
            early_stopping_rounds = 100
            if test_matrix:
                model = clf.train(params, train_matrix,num_round,valid_sets=test_matrix,
                                  early_stopping_rounds=early_stopping_rounds
                                  )
                pre= model.predict(te_x,num_iteration=model.best_iteration).reshape(-1,1)
                train[test_index]=pre
                test_pre[i, :]= model.predict(test_x, num_iteration=model.best_iteration).reshape(-1,1)
                cv_scores.append(mean_squared_error(te_y, pre)**0.5)

        elif clf_name in ["nn"]:
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.optimizers import SGD,RMSprop
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.utils import np_utils
            from keras.regularizers import l2
            from keras.models import Sequential
            clf = Sequential()
            clf.add(Dense(640, input_dim=tr_x.shape[1], activation="relu", W_regularizer=l2()))
            clf.add(Dropout(0.2))
            clf.add(Dense(640, activation="relu", W_regularizer=l2()))
            clf.add(Dropout(0.2))
            clf.add(Dense(1))
            clf.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=20)
            reduce = ReduceLROnPlateau(min_lr=0.0002,factor=0.05)
            clf.compile(optimizer="rmsprop", loss="mse")
            clf.fit(tr_x, tr_y,
                      batch_size=2560,
                      nb_epoch=100,
                      validation_data=[te_x, te_y],
                      #callbacks=[early_stopping, reduce]
                    )
            pre=clf.predict(te_x).reshape(-1,1)
            train[test_index]=pre
            test_pre[i,:]=clf.predict(test_x).reshape(-1,1)
            cv_scores.append(mean_squared_error(te_y, pre)**0.5)
        else:
            raise IOError("Please add new clf.")
        print("%s now score is:"%clf_name,cv_scores)
        with open("score.txt","a") as f:
            f.write("%s now score is:"%clf_name+str(cv_scores)+"\n")
    test[:]=test_pre.mean(axis=0)
    print("%s_score_list:"%clf_name,cv_scores)
    print("%s_score_mean:"%clf_name,np.mean(cv_scores))
    with open("score.txt", "a") as f:
        f.write("%s_score_mean:"%clf_name+str(np.mean(cv_scores))+"\n")
    return train.reshape(-1,1),test.reshape(-1,1)

def rf_reg(x_train, y_train, x_valid):
    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20, n_jobs=-1, random_state=2017, max_features="auto",verbose=1)
    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid,"rf")
    return rf_train, rf_test,"rf_reg"

def ada_reg(x_train, y_train, x_valid):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid,"ada")
    return ada_train, ada_test,"ada_reg"

def gb_reg(x_train, y_train, x_valid):
    gbdt = GradientBoostingRegressor(learning_rate=0.06, n_estimators=100, subsample=0.8, random_state=2017,max_depth=5,verbose=1)
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid,"gb")
    return gbdt_train, gbdt_test,"gb_reg"

def et_reg(x_train, y_train, x_valid):
    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=22, max_features="auto", n_jobs=-1, random_state=2017,verbose=1)
    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid,"et")
    return et_train, et_test,"et_reg"

def lr_reg(x_train, y_train, x_valid):
    lr_reg=LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, "lr")
    return lr_train, lr_test, "lr_reg"

def xgb_reg(x_train, y_train, x_valid):
    xgb_train, xgb_test = stacking_reg(xgboost, x_train, y_train, x_valid,"xgb")
    return xgb_train, xgb_test,"xgb_reg"

def lgb_reg(x_train, y_train, x_valid):
    lgb_train, lgb_test = stacking_reg(lightgbm, x_train, y_train, x_valid,"lgb")
    return lgb_train, lgb_test,"lgb_reg"

def nn_reg(x_train, y_train, x_valid):
    x_train=np.log10(x_train+1)
    x_valid=np.log10(x_valid+1)

    where_are_nan = np.isnan(x_train)
    where_are_inf = np.isinf(x_train)
    x_train[where_are_nan] = 0
    x_train[where_are_inf] = 0
    where_are_nan = np.isnan(x_valid)
    where_are_inf = np.isinf(x_valid)
    x_valid[where_are_nan] = 0
    x_valid[where_are_inf] = 0

    scale=StandardScaler()
    scale.fit(x_train)
    x_train=scale.transform(x_train)
    x_valid=scale.transform(x_valid)

    nn_train, nn_test = stacking_reg("", x_train, y_train, x_valid, "nn")
    return nn_train, nn_test, "nn_reg"
##########################################################################################################

#####################################################获取数据##############################################

###########################################################################################################
from create_data import get_data
if __name__=="__main__":
    np.random.seed(1)
    x_train, x_valid, y_train, train, test = get_data()

    train_id = train["item_id"].values
    test_id = test["item_id"].values

    folds = 5
    seed = 1
    kf = KFold(x_train.shape[0], n_folds=folds, shuffle=True, random_state=seed)

    #############################################选择模型###############################################
    #
    #
    #
    #clf_list = [xgb,nn,knn,gb,rf,et,lr,ada_reg,rf_reg,gb_reg,et_reg,xgb_reg,nn_reg]
    #clf_list = [xgb_reg,lgb_reg,nn_reg,lgb,xgb,lr,rf,et,gb,nn,knn]  #添加了magic的
    clf_list = [gb_reg]   #添加了magic的,补充三个reg
    #
    #
    column_list = []
    train_data_list=[]
    test_data_list=[]
    for clf in clf_list:
        train_data,test_data,clf_name=clf(x_train,y_train,x_valid)
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        if "reg" in clf_name:
            ind_num=1
        else:
            ind_num=3
        for ind in range(ind_num):
            column_list.append("%s_%s" % (clf_name, ind))

    train = np.concatenate(train_data_list, axis=1)
    test = np.concatenate(test_data_list, axis=1)

    train = pd.DataFrame(train)
    train.columns = column_list
    train["label"] = pd.Series(y_train)
    train["item_id"] = train_id

    test = pd.DataFrame(test)
    test.columns = column_list
    test["item_id"] = test_id

    train.to_csv("../stacking/stacking_train_gb.csv", index=None)
    test.to_csv("../stacking/stacking_test_gb.csv", index=None)


