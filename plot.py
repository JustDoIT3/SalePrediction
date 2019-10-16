# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('data/train_sales_data.csv')

    # 车型总销量曲线
    group_model_year_month_sale = df['salesVolume'].groupby([df['model'],df['bodyType'],df['regYear'],df['regMonth']])
    sum = group_model_year_month_sale.sum()
    print(sum)

    model = []
    sales = []
    x = range(24)
    for i in range(60):
        model.append(sum.index[i*24][0])
        sales.append(sum[sum.index[i*24][0]].tolist())
        plt.plot(x,sales[i])
        plt.title(model[i]+'-'+sum.index[i*24][1])
        plt.savefig('pic/'+str(model[i])+'-'+sum.index[i*24][1]+'.png')
        plt.show()

    print(model)
    print(sales)



