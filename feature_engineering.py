# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
seed = 7
np.random.seed(seed)

from keras.models import Sequential,Model,load_model
from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten

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

    df = pd.read_csv('data/correct_train_sales_data.csv')
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
    # single_num = 15840
    # user_search = pd.read_csv('user_search.csv')
    # train_user_search = pd.DataFrame(index = list(range(single_num)))
    # train_user_search['popularity'] = user_search['popularity'].iloc[single_num:single_num*2].values
    # train_user_search['carCommentVolum'] = user_search['carCommentVolum'].iloc[single_num:single_num*2].values
    # train_user_search['newsReplyVolum'] = user_search['newsReplyVolum'].iloc[single_num:single_num*2].values
    # train_df = pd.concat([train_df, train_user_search], axis=1)
    
    train_df.to_csv('data/train1021.csv',index=None)

    print(train_df)

# CNN自动提取特征
def CNN_feature():

    # 模型搭建 前12个月销量作为输入
    model = Sequential()
    model.add(Conv1D(50,5,activation='relu',input_shape=(12,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(100,3,activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(1))

    # 数据集准备 模型训练
    train_df = pd.read_csv('data/train.csv')
    train_x = train_df.iloc[:,6:18].values
    train_y = train_df.iloc[:,0].values
    train_x = train_x.reshape((15840,12,1))
    # 数据集shuffle
    index=np.arange(15840)
    np.random.shuffle(index)
    train_shuffle_x = train_x[index,:,:]
    train_shuffle_y = train_y[index]

    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
    history = model.fit(train_shuffle_x,train_shuffle_y,epochs=1000,batch_size=15840,validation_split=0.2,shuffle=False)

    # 取出中间层特征输出模型存储
    feature_model = Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)
    feature_model.save('data/feature_model.h5')
    cnn_feature = feature_model.predict(train_x)
    for i in range(100):
        train_df['cnnfeature'+str(i+1)] = cnn_feature[:,i]

    train_df.to_csv('data/train_cnn.csv',index=None)


    # rmse = history.history['val_mean_squared_error']
    # plt.plot(range(1500),rmse)
    # plt.show()
    # print(rmse.index(min(rmse)))

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
    # cnn特征
    model = load_model('data/feature_model.h5')
    cnn_feature = np.zeros((22*60*4,100))
    x = evaluation_df.iloc[:22*60,5:17].values
    x = x.reshape((1320,12,1))
    predict = model.predict(x)
    cnn_feature[:22*60,:] = predict
    for i in range(100):
        evaluation_df['cnnfeature'+str(i+1)] = cnn_feature[:,i]

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
    # single_num = 15840
    # user_search = pd.read_csv('user_search.csv')
    # evaluation_user_search = pd.DataFrame(index = list(range(5280)))
    # evaluation_user_search['popularity'] = user_search['popularity'].iloc[single_num*2:].values
    # evaluation_user_search['carCommentVolum'] = user_search['carCommentVolum'].iloc[single_num*2:].values
    # evaluation_user_search['newsReplyVolum'] = user_search['newsReplyVolum'].iloc[single_num*2:].values
    # evaluation = pd.concat([evaluation_df, evaluation_user_search], axis=1)

    evaluation_df.to_csv('data/evaluation_cnn.csv',index=None)

    print(evaluation_df)


# 销量序列的异常检测和修正
def correct_train_data():
    df = pd.read_csv('data/train_sales_data.csv')
    # 省份 车型 1-12月销量增长情况列表
    growth_df = pd.DataFrame(index = list(range(22*60)))
    growth_df['province'] = df.iloc[:22*60,1].values
    growth_df['model'] = df.iloc[:22*60,2].values
    growth = df.iloc[22*60*12:,6].values - df.iloc[:22*60*12,6].values
    growth_arr = np.zeros((22*60,12))
    for i in range(12):
        growth_arr[:,i] += growth[22*60*i:22*60*(i+1)]

    # 箱型图求每个省份车型增长异常数据 在哪个月份出现异常
    growth_list = []
    Q1 = []
    Q3 = []
    for i in range(22*60):
        growth_list.append(growth_arr[i,:].tolist())
        Q1.append(np.percentile(growth_arr[i,:], 25))
        Q3.append(np.percentile(growth_arr[i,:], 75))
    # 箱型图计算
    growth_df['growth'] = growth_list
    IQR = np.array(Q3) - np.array(Q1)
    lower_quartile = np.array(Q1) - 1.5*IQR
    upper_quartile = np.array(Q3) + 1.5*IQR
    # 异常月份 即离群点检测
    abnormal_month = []
    for i in range(22*60):
        abnormal_month.append([index+1 for index,value in enumerate(growth_list[i]) if value<lower_quartile[i] or value>upper_quartile[i]])

    growth_df['lowerQuartile'] = lower_quartile
    growth_df['upperQuartile'] = upper_quartile
    growth_df['abnormal'] = abnormal_month

    map = {}
    for i in range(12):
        map[i+1] = 0
    for month_list in abnormal_month:
        for month in month_list:
            map[month] += 1
    print(map)


    # 使用箱型图的上界或下界对异常销量值进行修正
    count16 = 0
    count17 = 0
    for province_model in range(22*60):
        for month in abnormal_month[province_model]:
            if growth_list[province_model][month-1] < lower_quartile[province_model]:
                correct = lower_quartile[province_model]
            else:
                correct = upper_quartile[province_model]
            # 判断修正16年or17年 修正与周围数据的差距大的那一年
            data = df.loc[(df.adcode == growth_df.iloc[province_model,0]) & (df.model == growth_df.iloc[province_model,1])
                          & (df.regMonth == month)]['salesVolume']
            data16 = data.iloc[0]
            data17 = data.iloc[1]
            if month == 1:
                compare = df.loc[(df.adcode == growth_df.iloc[province_model,0]) & (df.model == growth_df.iloc[province_model,1])
                                 & (df.regMonth == 2)]['salesVolume']
            else:
            # if month != 1 and month != 12:
                compare = df.loc[(df.adcode == growth_df.iloc[province_model,0]) & (df.model == growth_df.iloc[province_model,1])
                                 & (df.regMonth == month-1)]['salesVolume']
            compare16 = compare.iloc[0]
            compare17 = compare.iloc[1]

            if abs(compare16 - data16) > abs(compare17 - data17):
                # 修正16年数据
                df.loc[(df.adcode == growth_df.iloc[province_model,0]) & (df.model == growth_df.iloc[province_model,1])
                       & (df.regMonth == month) & (df.regYear == 2016),'salesVolume'] = math.ceil(data17 - correct)
                count16 += 1
            else:
                # 修正17年数据
                df.loc[(df.adcode == growth_df.iloc[province_model,0]) & (df.model == growth_df.iloc[province_model,1])
                       & (df.regMonth == month) & (df.regYear == 2017),'salesVolume'] = math.ceil(data16 + correct)
                count17 += 1

    print(count16)
    print(count17)
    growth_df.to_csv('data/growth.csv',index=None)
    df.to_csv('data/correct_train_sales_data.csv',index=None)


if __name__ == '__main__':


    # correct_train_data()
    # train_feature()
    evaluation_feature()
    # CNN_feature()



