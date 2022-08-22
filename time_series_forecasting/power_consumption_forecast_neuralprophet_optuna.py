# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/8/22 16:52
# @Author : liumin
# @File : power_consumption_forecast_neuralprophet_optuna.py

from copy import deepcopy
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")

df_train = pd.read_csv("shenghuoqu0815.csv")
df_test = pd.read_csv("shenghuoqu0821.csv")
print(df_train.info())
print(df_test.info())

n_lags = 12
n_forecasts=12*1

default_para = dict(n_lags = n_lags,n_forecasts= n_forecasts,changepoints_range=0.8,n_changepoints=5, trend_reg=0.1,normalize='minmax',learning_rate =1)

param_types = dict(changepoints_range='float',n_changepoints='int',trend_reg='float',learning_rate='float',month_order = 'int',week_order ='int')
bounds = {'changepoints_range': [0.6,0.8,0.9],
          'n_changepoints': [4, 8],
          'trend_reg': [0.001, 1],
          'learning_rate': [0.001, 1],
          'day_order': [1, 7],
         }

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

def nph_warper(trial,ts):
    params = {}
    params['changepoints_range'] = trial.suggest_categorical('changepoints_range', bounds['changepoints_range'])
    params['n_changepoints'] = trial.suggest_int('n_changepoints', bounds['n_changepoints'][0], bounds['n_changepoints'][1])
    params['trend_reg'] = trial.suggest_loguniform('trend_reg', bounds['trend_reg'][0], bounds['trend_reg'][1])
    params['learning_rate'] = trial.suggest_loguniform('learning_rate', bounds['learning_rate'][0], bounds['learning_rate'][1])
    # params['month_order'] = trial.suggest_int('month_order', bounds['month_order'][0], bounds['month_order'][1])
    # params['week_order'] = trial.suggest_int('week_order', bounds['week_order'][0], bounds['week_order'][1])
    params['day_order'] = trial.suggest_int('day_order', bounds['day_order'][0], bounds['day_order'][1])
    temp_para = deepcopy(default_para)
    temp_para.update(params)
    METRICS = ['SmoothL1Loss', 'MAE', 'RMSE']
    metrics_test = pd.DataFrame(columns=METRICS)
    m = create_nph(**temp_para)
    folds = m.crossvalidation_split_df(ts, freq="H", k=5, fold_pct=0.2, fold_overlap_pct=0.5)
    for df_train, df_test in folds:
        m = create_nph(**temp_para)
        train = m.fit(df_train)
        test = m.test(df=df_test)
        metrics_test = metrics_test.append(test[METRICS].iloc[-1])
    out = metrics_test['MAE'].mean()
    return out

def objective(trial):
    ts = df_train.copy() #select column associated with region
    return nph_warper(trial,ts)


study = optuna.create_study(direction='minimize',study_name='nph_tuning3', load_if_exists=True, storage="sqlite:///nph.db")
study.optimize(objective, n_trials=5)
print(study.best_trial)
print(study.best_params)

best_para = deepcopy(default_para)
best_para.update(study.best_params)

m = create_nph(**best_para)


# weekly_seasonality=True,
metrics = m.fit(df_train, freq='5min')
forecast_train = m.predict(df_train, decompose=False)
forecast_test = m.predict(df_test, decompose=False)

fig = m.plot(forecast_train)
fig = m.plot(forecast_test)
fig_param = m.plot_parameters()

plt.show()