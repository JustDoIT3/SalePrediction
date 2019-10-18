# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import math

# 省份 车型 有序编码
def encode_map(df):
    province = df['adcode'][:22]
    model = df['model'].unique()
    bodyType = df['bodyType'].unique()
    province_map = {}
    model_map = {}
    bodyType_map = {}
    for i in range(1,23):
        province_map[province[i-1]] = i
    for i in range(60):
        model_map[model[i]] = i+1
    for i in range(len(bodyType)):
        bodyType_map[bodyType[i]] = i+1
    return province_map,model_map,bodyType_map

def model_bodytype_map(df):
    map = {}
    for i in range(60):
        map[df.iloc[i*22,2]] = df.iloc[i*22,3]
    return map

def train_feature():
    num_province = 22
    num_model = 60

    start_label = 22*60*12

    df = pd.read_csv('data/train_sales_data.csv')
    df_search = pd.read_csv('data/train_search_data.csv')
    df_merge = pd.merge(df,df_search)

    train_df = pd.DataFrame(index = list(range(start_label)))

    train_df['label'] = df.iloc[start_label:,6].values
    train_df['province'] = df.iloc[start_label:,1].values
    train_df['model'] = df.iloc[start_label:,2].values
    train_df['bodyType'] = df.iloc[start_label:,3].values
    train_df['month'] = df.iloc[start_label:,5].values
    is_spring_festival = np.zeros((start_label,1),dtype='uint8')
    is_spring_festival[:num_province*num_model] = 1
    train_df['isSpringFestival'] = is_spring_festival
    for i in range(12):
        train_df['sales'+str(12-i)+'MonthAgo'] = df.iloc[22*60*i:22*60*(i+12),6].values
    train_df['averageSales2Month'] = round((train_df['sales1MonthAgo']+train_df['sales2MonthAgo'])/2)
    train_df['averageSales3Month'] = round((train_df['sales1MonthAgo']+train_df['sales2MonthAgo']+train_df['sales3MonthAgo'])/3)
    train_df['firstOrderTrend1'] = train_df['sales1MonthAgo'] - train_df['sales2MonthAgo']
    train_df['firstOrderTrend2'] = train_df['sales2MonthAgo'] - train_df['sales3MonthAgo']
    train_df['secondOrderTrend'] = train_df['firstOrderTrend1'] - train_df['firstOrderTrend2']

    # 省份 车型 one-hot编码
    # province_onehot = pd.get_dummies(train_df['province'])
    # model_onehot = pd.get_dummies(train_df['model'])
    # train_df = pd.concat([train_df,province_onehot,model_onehot],axis=1)
    # train_df = train_df.drop(['province','model'],axis=1)
    # 省份 车型 按序编码
    province_map,model_map,bodyType_map = encode_map(df)
    train_df['province'] = train_df['province'].map(province_map)
    train_df['model'] = train_df['model'].map(model_map)
    train_df['bodyType'] = train_df['bodyType'].map(bodyType_map)
   
    # 融合媒体数据
    single_num = 15840
    user_search = pd.read_csv('user_search.csv')
    train_user_search = pd.DataFrame(index = list(range(single_num)))
    train_user_search['popularity'] = user_search['popularity'].iloc[single_num:single_num*2].values
    train_user_search['carCommentVolum'] = user_search['carCommentVolum'].iloc[single_num:single_num*2].values
    train_user_search['newsReplyVolum'] = user_search['newsReplyVolum'].iloc[single_num:single_num*2].values
    train_df = pd.concat([train_df, train_user_search], axis=1)
    
    train_df.to_csv('data/train.csv',index=None)

    print(train_df)

def evaluation_feature():
    sale_df = pd.read_csv('data/train_sales_data.csv')
    df = pd.read_csv('data/evaluation_public.csv')

    evaluation_df = pd.DataFrame(index=list(range(22*60*4)))

    evaluation_df['province'] = df.iloc[:,2]
    evaluation_df['model'] = df.iloc[:,3]
    evaluation_df['bodyType'] = evaluation_df['model'].map(model_bodytype_map(sale_df))
    evaluation_df['month'] = df.iloc[:,5]
    is_spring_festival = np.zeros((22*60*4,1),dtype='uint8')
    is_spring_festival[22*60:22*60*2] = 1
    evaluation_df['isSpringFestival'] = is_spring_festival
    for i in range(9):
        evaluation_df['sales'+str(12-i)+'MonthAgo'] = sale_df.iloc[22*60*(12+i):22*60*(i+12+4),6].values
    for i in range(3,0,-1):
        sales = np.zeros((22*60*4,1))
        sales[:22*60*i,0] = sale_df.iloc[22*60*(24-i):,6].values
        evaluation_df['sales'+str(i)+'MonthAgo'] = sales
    evaluation_df['averageSales2Month'] = round((evaluation_df['sales1MonthAgo']+evaluation_df['sales2MonthAgo'])/2)
    evaluation_df['averageSales3Month'] = round((evaluation_df['sales1MonthAgo']+evaluation_df['sales2MonthAgo']+evaluation_df['sales3MonthAgo'])/3)
    evaluation_df['firstOrderTrend1'] = evaluation_df['sales1MonthAgo'] - evaluation_df['sales2MonthAgo']
    evaluation_df['firstOrderTrend2'] = evaluation_df['sales2MonthAgo'] - evaluation_df['sales3MonthAgo']
    evaluation_df['secondOrderTrend'] = evaluation_df['firstOrderTrend1'] - evaluation_df['firstOrderTrend2']

    # 省份 车型 one-hot编码
    # province_onehot = pd.get_dummies(df['adcode'])
    # model_onehot = pd.get_dummies(df['model'])
    # evaluation_df = pd.concat([evaluation_df,province_onehot,model_onehot],axis=1)
    # 省份 车型 按序编码
    province_map,model_map,bodyType_map = encode_map(sale_df)
    evaluation_df['province'] = evaluation_df['province'].map(province_map)
    evaluation_df['model'] = evaluation_df['model'].map(model_map)
    evaluation_df['bodyType'] = evaluation_df['bodyType'].map(bodyType_map)
    
    # 融合媒体数据
    single_num = 15840
    user_search = pd.read_csv('user_search.csv')
    evaluation_user_search = pd.DataFrame(index = list(range(5280)))
    evaluation_user_search['popularity'] = user_search['popularity'].iloc[single_num*2:].values
    evaluation_user_search['carCommentVolum'] = user_search['carCommentVolum'].iloc[single_num*2:].values
    evaluation_user_search['newsReplyVolum'] = user_search['newsReplyVolum'].iloc[single_num*2:].values
    evaluation = pd.concat([evaluation_df, evaluation_user_search], axis=1)

    evaluation.to_csv('data/evaluation.csv',index=None)

    print(evaluation_df)

if __name__ == '__main__':

    # train_feature()
    evaluation_feature()
