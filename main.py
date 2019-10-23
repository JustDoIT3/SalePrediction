# -*- coding: UTF-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.models import load_model

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def predict_sales_online():
    train_df = pd.read_csv('data/train_cnn.csv')

    n_estimators = 847
    model = xgb.XGBRegressor(
        max_depth=5 , learning_rate=0.05, n_estimators=n_estimators,
        objective='reg:gamma', tree_method = 'hist',subsample=0.9,
        colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse'
    )

    # # n_estimators = 847 1374
    # # k折交叉验证 用于调参 训练12个模型
    # test_loss = []
    # kf = KFold(n_splits=12)
    # for train_index , test_index in kf.split(train_df):
    #     train_x = train_df.drop(['label'],axis=1).iloc[train_index]
    #     train_y = train_df['label'][train_index]
    #     test_x = train_df.drop(['label'],axis=1).iloc[test_index]
    #     test_y = train_df['label'][test_index]
    #
    #     model.fit(train_x,train_y,eval_set=[(test_x,test_y)])
    #
    #     result = model.evals_result()
    #     loss = result['validation_0']['rmse']
    #     test_loss.append(loss)
    #
    # # 每轮中所有折rmse平均值
    # average_rmse = [np.mean([x[i] for x in test_loss]) for i in range(n_estimators)]
    # plt.plot(range(n_estimators),average_rmse)
    # plt.show()
    # print(average_rmse.index(min(average_rmse)))


    # 按最佳训练轮次训练全部数据集

    # 划分训练集和测试集
    train_x = train_df.drop(['label'],axis=1)
    train_y = train_df['label']

    model.fit(train_x,train_y,eval_set=[(train_x, train_y)])

    # 训练集和测试集的rmse曲线
    # results = model.evals_result()
    # train_loss = results['validation_0']['rmse']
    # plt.plot(range(len(train_loss)),train_loss,label='train')
    # plt.show()

    # 预测提交结果
    evaluation_df = pd.read_csv('data/evaluation_cnn.csv')
    submit = pd.read_csv('data/submit_example.csv')
    prediction = []
    cnn_model = load_model('data/feature_model.h5',compile=False)
    # 对1-4月逐月预测
    for i in range(4):
        # 将之前月份的预测结果加入特征
        for j in range(len(prediction)):
            evaluation_df.iloc[22*60*i:22*60*(i+1),17-(i-j)] = prediction[j]
        evaluation_df['averageSales2Month'] = round((evaluation_df['sales1MonthAgo']+evaluation_df['sales2MonthAgo'])/2)
        evaluation_df['averageSales3Month'] = round((evaluation_df['sales1MonthAgo']+evaluation_df['sales2MonthAgo']+evaluation_df['sales3MonthAgo'])/3)
        evaluation_df['firstOrderTrend1'] = evaluation_df['sales1MonthAgo'] - evaluation_df['sales2MonthAgo']
        evaluation_df['firstOrderTrend2'] = evaluation_df['sales2MonthAgo'] - evaluation_df['sales3MonthAgo']
        evaluation_df['secondOrderTrend'] = evaluation_df['firstOrderTrend1'] - evaluation_df['firstOrderTrend2']
        # cnn特征计算
        if i != 0:
            x = evaluation_df.iloc[22*60*i:22*60*(i+1),5:17].values
            x = x.reshape((1320,12,1))
            predict = cnn_model.predict(x)
            evaluation_df.iloc[22*60*i:22*60*(i+1),22:] = predict
        # 预测各月
        evaluation_x = evaluation_df.iloc[22*60*i:22*60*(i+1),:]
        ans = model.predict(evaluation_x)
        submit.iloc[22*60*i:22*60*(i+1),1] = np.ceil(ans)
        prediction.append(np.ceil(ans))
        # 将本月合并到训练集进行训练
        # print('training '+str(i)+' round')
        # train_x.append(test_x)
        # train_y.append(test_y)
        # test_x = evaluation_x
        # test_y = pd.DataFrame(np.ceil(ans))
        # model.fit(train_x,train_y,eval_set=[(train_x, train_y),(test_x,test_y)],early_stopping_rounds=100)


    # 查看特征是否被正确计算
    evaluation_df.to_csv('data/evaluation1.csv',index=None)
    submit.to_csv('data/1023.csv',index=None)

def predict_sales_offline():
    df = pd.read_csv('data/train.csv')
    correct_df = pd.read_csv('data/train1021.csv')
    train_df = correct_df.iloc[:22*60*8]

    n_estimators = 539
    model = xgb.XGBRegressor(
        max_depth=5 , learning_rate=0.05, n_estimators=n_estimators,
        objective='reg:gamma', tree_method = 'hist',subsample=0.9,
        colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse'
    )

    # # n_estimators = 847 1374
    # # k折交叉验证 用于调参 训练12个模型
    # test_loss = []
    # kf = KFold(n_splits=8)
    # for train_index , test_index in kf.split(train_df):
    #     train_x = train_df.drop(['label'],axis=1).iloc[train_index]
    #     train_y = train_df['label'][train_index]
    #     test_x = train_df.drop(['label'],axis=1).iloc[test_index]
    #     test_y = train_df['label'][test_index]
    #
    #     model.fit(train_x,train_y,eval_set=[(test_x,test_y)])
    #
    #     result = model.evals_result()
    #     loss = result['validation_0']['rmse']
    #     test_loss.append(loss)
    #
    # # 每轮中所有折rmse平均值
    # average_rmse = [np.mean([x[i] for x in test_loss]) for i in range(n_estimators)]
    # plt.plot(range(n_estimators),average_rmse)
    # plt.show()
    # print(average_rmse.index(min(average_rmse)))


    # 按最佳训练轮次训练全部数据集

    # 划分训练集和测试集
    train_x = train_df.drop(['label'],axis=1)
    train_y = train_df['label']

    model.fit(train_x,train_y,eval_set=[(train_x, train_y)])

    # 训练集和测试集的rmse曲线
    # results = model.evals_result()
    # train_loss = results['validation_0']['rmse']
    # plt.plot(range(len(train_loss)),train_loss,label='train')
    # plt.show()

    # 预测提交结果
    evaluation_df = correct_df.drop(['label'],axis=1).iloc[22*60*8:]
    submit = pd.read_csv('data/submit_example.csv')
    prediction = []
    # 对1-4月逐月预测
    for i in range(4):
        # 将之前月份的预测结果加入特征
        for j in range(len(prediction)):
            evaluation_df.iloc[22*60*i:22*60*(i+1),17-(i-j)] = prediction[j]
        evaluation_df['averageSales2Month'] = round((evaluation_df['sales1MonthAgo']+evaluation_df['sales2MonthAgo'])/2)
        evaluation_df['averageSales3Month'] = round((evaluation_df['sales1MonthAgo']+evaluation_df['sales2MonthAgo']+evaluation_df['sales3MonthAgo'])/3)
        evaluation_df['firstOrderTrend1'] = evaluation_df['sales1MonthAgo'] - evaluation_df['sales2MonthAgo']
        evaluation_df['firstOrderTrend2'] = evaluation_df['sales2MonthAgo'] - evaluation_df['sales3MonthAgo']
        evaluation_df['secondOrderTrend'] = evaluation_df['firstOrderTrend1'] - evaluation_df['firstOrderTrend2']
        # 预测各月
        evaluation_x = evaluation_df.iloc[22*60*i:22*60*(i+1),:]
        ans = model.predict(evaluation_x)
        submit.iloc[22*60*i:22*60*(i+1),1] = np.ceil(ans)
        prediction.append(np.ceil(ans))
        # 将本月合并到训练集进行训练
        # print('training '+str(i)+' round')
        # train_x.append(test_x)
        # train_y.append(test_y)
        # test_x = evaluation_x
        # test_y = pd.DataFrame(np.ceil(ans))
        # model.fit(train_x,train_y,eval_set=[(train_x, train_y),(test_x,test_y)],early_stopping_rounds=100)


    # 预测rmse
    submit.to_csv('data/1021.csv',index=None)
    print(rmse(submit.iloc[:,1].values,df['label'][22*60*8:].values))



if __name__ == '__main__':

    predict_sales_online()



