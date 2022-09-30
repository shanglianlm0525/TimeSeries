# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/8/18 9:06
# @Author : liumin
# @File : power_consumption_forecast_neuralprophet.py

from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")

df_train = pd.read_csv("data/shenghuoqu0815.csv")
df_test = pd.read_csv("data/shenghuoqu0821.csv")
print(df_train.info())
print(df_test.info())


def create_nph(**para):
 temp_para = deepcopy(para)
 day_order = temp_para.pop('day_order')
 # month_order = temp_para.pop('month_order')
 # week_order = temp_para.pop('week_order')
 m = NeuralProphet(**temp_para)
 # m = m.add_seasonality('my_month', 28, month_order)
 # m = m.add_seasonality('my_week', 7, week_order)
 m = m.add_seasonality('my_day', 24, day_order)
 return m


n_lags = 12
n_forecasts=12*1

default_para = dict(n_lags = n_lags,n_forecasts= n_forecasts,changepoints_range=0.8,n_changepoints=5, trend_reg=0.1,normalize='minmax',learning_rate =1)
best_para = {'changepoints_range': 0.6, 'day_order': 1, 'learning_rate': 0.005463461412824059,
             'n_changepoints': 7, 'trend_reg': 0.006287378033976726}
default_para.update(best_para)
print(default_para)
m = create_nph(**default_para)
'''
m = NeuralProphet(n_lags = n_lags,n_forecasts= n_forecasts,
                  yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True,
                  normalize='minmax')
'''
# weekly_seasonality=True,
metrics = m.fit(df_train, freq='5min')
forecast_train = m.predict(df_train)
forecast_test = m.predict(df_test)

fig = m.plot(forecast_train)
fig = m.plot(forecast_test)
fig_param = m.plot_parameters()

plt.show()