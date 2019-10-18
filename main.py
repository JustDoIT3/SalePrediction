# -*- coding: UTF-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    train_df = pd.read_csv('data/train.csv')

    model = xgb.XGBRegressor(
        max_depth=5 , learning_rate=0.05, n_estimators=2000,
        objective='reg:gamma', tree_method = 'hist',subsample=0.9,
        colsample_bytree=0.7, min_child_samples=5,eval_metric = 'rmse'
    )

    # 划分训练集和测试集
    train_x = train_df.drop(['label'],axis=1).iloc[:22*60*11,:]
    train_y = train_df['label'][:22*60*11]
    test_x = train_df.drop(['label'],axis=1).iloc[22*60*11:,:]
    test_y = train_df['label'][22*60*11:]

    model.fit(train_x,train_y,eval_set=[(train_x, train_y),(test_x,test_y)],early_stopping_rounds=100)

    # 训练集和测试集的rmse曲线
    results = model.evals_result()
    train_loss = results['validation_0']['rmse']
    test_loss = results['validation_1']['rmse']
    plt.plot(range(len(train_loss)),train_loss,label='train')
    plt.plot(range(len(test_loss)),test_loss,label='test')
    plt.show()
    # 预测
    ans = model.predict(test_x)


    # 预测提交结果
    evaluation_df = pd.read_csv('data/evaluation.csv')
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

    # 查看特征是否被正确计算
    evaluation_df.to_csv('data/evaluation1.csv',index=None)
    submit.to_csv('data/1018.csv')



