# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/8/18 9:06
# @Author : liumin
# @File : power_consumption_forecast_neuralprophet.py

import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")

df_train = pd.read_csv("data/shenghuoqu0815.csv")
df_test = pd.read_csv("data/shenghuoqu0821.csv")
print(df_train.info())
print(df_test.info())

n_lags = 12
n_forecasts=12*1
# n_lags = n_lags,n_forecasts= n_forecasts,
# changepoints_range=0.85, n_changepoints=20,
m = NeuralProphet(n_lags = n_lags,n_forecasts= n_forecasts,
                  yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True,
                  normalize='minmax')
# weekly_seasonality=True,
metrics = m.fit(df_train, freq='5min')
forecast_train = m.predict(df_train)
forecast_test = m.predict(df_test)

fig = m.plot(forecast_train)
fig = m.plot(forecast_test)
fig_param = m.plot_parameters()

plt.show()