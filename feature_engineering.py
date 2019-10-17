# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import math

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
    province_onehot = pd.get_dummies(train_df['province'])
    model_onehot = pd.get_dummies(train_df['model'])
    train_df = pd.concat([train_df,province_onehot,model_onehot],axis=1)
    train_df = train_df.drop(['province','model'],axis=1)

    train_df.to_csv('data/train.csv')

    print(train_df)

if __name__ == '__main__':

    sale_df = pd.read_csv('data/train_sales_data.csv')
    df = pd.read_csv('data/evaluation_public.csv')

    evaluation_df = pd.DataFrame(index=list(range(22*60*4)))

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

    province_onehot = pd.get_dummies(df['adcode'])
    model_onehot = pd.get_dummies(df['model'])
    evaluation_df = pd.concat([evaluation_df,province_onehot,model_onehot],axis=1)

    evaluation_df.to_csv('data/evaluation.csv')

    print(evaluation_df)