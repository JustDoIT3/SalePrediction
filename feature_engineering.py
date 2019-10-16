# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import math

if __name__ == '__main__':

    num_province = 22
    num_model = 60

    start_label = 22*60*12

    df = pd.read_csv('data/train_sales_data.csv')
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

    train_df.to_csv('data/train.csv')


    print(train_df)