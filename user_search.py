#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
import warnings
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# 利用前两年的数据预测媒体数据,使用之前没有利用媒体数据的模型预测出来的销量数据进行填充
def user_search_feature():
    t_sales = pd.read_csv('data/train_sales_data.csv')
    t_search = pd.read_csv('data/train_search_data.csv')
    t_user   = pd.read_csv('data/train_user_reply_data.csv')
    eva = pd.read_csv('data/evaluation_public.csv')
    # 把所有数据进行融合
    data = pd.concat([t_sales, eva], ignore_index=True)
    data = data.merge(t_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
    data = data.merge(t_user, 'left', on=['model', 'regYear', 'regMonth'])
    data['bodyType'] = data['model'].map(t_sales.drop_duplicates('model').set_index('model')['bodyType']) #完善预测数据的车型
    data['id'] = np.array(range(0,36960))
    #LabelEncoder
    for i in ['bodyType', 'model','province']:
        data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
    data['month'] = (data['regYear'] - 2016) * 12 + data['regMonth']
    #data['bodyType'] = data['bodyType'].astype('str')
    #data['model'] = data['model'].astype('str')
    # 添加春节标识
    num_province = 22
    num_model = 60
    single_num = 15840
    total_num = 36960   # 前两年数据共计31680条，每年15840条，需要预测数据5280条
    spring_num = 22*60*1
    is_spring_festival = np.zeros((total_num,1),dtype='uint8')
    is_spring_festival[:num_province*num_model] = 1
    is_spring_festival[single_num : single_num + num_province*num_model] = 1
    is_spring_festival[single_num : single_num*2 + num_province*num_model] = 1
    data['isSpringFestival'] = is_spring_festival
    isSpringFestival_bool = data['isSpringFestival'].astype('bool')
    data['isSpringFestival'] = isSpringFestival_bool
    # 删除省份编号
    data = data.drop('adcode',axis = 1)
    # 采用第一次训练出来的预测销量数据填充最后四个月的销量数据
    salesVolume_1 = pd.read_csv('data/1018.csv')
    salesVolume_1 = salesVolume_1['forecastVolum']
    fill_data['salesVolume'].iloc[single_num*2:] = salesVolume_1.values

# 预测出'popularity','newsReplyVolum','carCommentVolum'三组数据   
def user_search_model():
    train_data = fill_data[0:single_num*2]
    pre_data = fill_data[single_num*2:]
    features = ['bodyType','id','model','province','regMonth','regYear','salesVolume','month','isSpringFestival']
    train = train_data[features]
    X_train,X_test,y_train,y_test =train_test_split(train,target,test_size=0.3)
    for col in ['popularity','newsReplyVolum','carCommentVolum']:
        targets = [col]
        target = train_data[targets]
        model = xgb.XGBRegressor(
                max_depth=5 , learning_rate=0.05, n_estimators=2000,
                objective='reg:gamma', tree_method = 'hist',subsample=0.9,
                colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse'
            )
        model.fit(X_train,y_train,eval_set=[(X_train, y_train),(X_test,y_test)],early_stopping_rounds=10)
        pre = pre_data[features]
        x = model.predict(pre)
        fill_data[col].iloc[single_num*2:] = x.astype(int)
    fill_data.to_csv("data/user_search.csv",index = False)

