#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from clear_sky_irradiance import clear_sky_model
from clear_date_index import generate_clear_index_new
from fit_model import fit_model
from forecast import series_to_supervised, forecast_

# def main_cs():
data_root = os.path.abspath("/Users/mayuan/Downloads/projects/data_jilin")

# basic info
df = pd.read_excel(r'basic_info.xlsx', sheet_name='solar')

k = 0
latitude = df['Lat'][k]
lontitude = df['Lon'][k]
station_name = str(df['ID'][k]) + '.csv'
nwp_name = df['NWP_ID'][k] + '.csv'

dir_ = os.path.join(data_root, 'data')
filepath_power = os.path.join(dir_, 'Power', station_name)
filepath_nwp = os.path.join(dir_, 'NWP', nwp_name)

# load data
power = pd.read_csv(filepath_power, index_col=0, parse_dates=True)
nwp = pd.read_csv(filepath_nwp, index_col=0, parse_dates=True)

# calculate clear sky irradiannce
E = clear_sky_model(latitude, lontitude)

# process nwp
T_amb1 = nwp.temperature[-33:].copy()
T_amb2 = nwp.temperature[:-33].copy()
T_amb = T_amb1.append(T_amb2)
T_amb = pd.DataFrame({'T_amb': T_amb.values.reshape(-1)},
                     index=power.index) - 274.15

# use data of 2018
NOCT = 45.5  # from reference
T_amb_2018 = T_amb[96 * 365:]
T_c = T_amb_2018.values.reshape(-1) + E.values.reshape(-1) * (NOCT - 20) / 800
K_Tc_2018 = 1 - 0.005 * (T_c[:365 * 96] - 25)
power_2018 = power[96 * 365:]

# calculate clear index
cap = power.max().tolist()[0]
clear_set, _ = generate_clear_index_new(power_2018, cap)

####################################################################################
# clear_set = [227, 251, 252, 258, 262, 269, 274, 275, 276, 283, 284, 287, 288, 289, 290, 291, 304, 308, 309, 323, 324, 325, 328, 332, 337, 340, 341]
#####################################################################################
# calculate P_fit
P_fit, df_parameter, ratio_test, _ = fit_model(power_2018, E, K_Tc_2018, clear_set)
#
# # forecast
# start_time_k = 20
# y_hat_all_short, y_true_all, RMSE_ = forecast_(
#     E, P_fit, power_2018, clear_set, ratio_test, K_Tc_2018, 20)
#
# y_hat_all_long, y_true_all, RMSE_ = forecast_(
#     E, P_fit, power_2018, clear_set, ratio_test, K_Tc_2018, 28)
#
# y_hat_final_1 = y_hat_all_short.iloc[:,:3]
# y_hat_final_2 = y_hat_all_long.iloc[:,3:]
# y_hat_final = pd.concat([y_hat_final_1, y_hat_final_2], axis=1)
    # return y_hat_all, y_true_all, RMSE_

# forecast for 2019
power_2019 = pd.read_csv('../data_jilin/data_2019_new/solar/501.csv', index_col=0, parse_dates=True)
power_2019 = power_2019[:-1]
nwp_2019 = pd.read_csv('../data_jilin/data_2019_new/nwp_2019/'+nwp_name, index_col=0, parse_dates=True)
# process nwp
T_amb_1 = nwp.temperature[-33:].copy()
T_amb_2019_2 = nwp_2019.temperature[:-33].copy()
T_amb_2019 = T_amb1.append(T_amb_2019_2)
T_amb_2019 = pd.DataFrame({'T_amb': T_amb_2019.values.reshape(-1)},
                     index=power_2019.index) - 274.15

# use data of 2018
NOCT = 45.5  # from reference
T_c_2019 = T_amb_2019.values.reshape(-1) + E[:len(nwp_2019)].values.reshape(-1) * (NOCT - 20) / 800
K_Tc_2019 = 1 - 0.005 * (T_c_2019- 25)

clear_set_2019, _ = generate_clear_index_new(power_2019, cap)

# forecast
start_time_k = 20
y_hat_all_short_2019, y_true_all_2019, RMSE_ = forecast_(
    E, P_fit, power_2019, clear_set_2019, ratio_test, K_Tc_2018, 20)

y_hat_all_long_2019, y_true_all_2019, RMSE_ = forecast_(
    E, P_fit, power_2019, clear_set_2019, ratio_test, K_Tc_2018, 28)

y_hat_final_1_2019 = y_hat_all_short_2019.iloc[:,:3]
y_hat_final_2_2019 = y_hat_all_long_2019.iloc[:,3:]
y_hat_final_2019 = pd.concat([y_hat_final_1_2019, y_hat_final_2_2019], axis=1)
y_hat_final_2019.to_csv('y_hat_final_.csv')
#
# # plot
# import datetime
# date_ = datetime.date(2019,2,12)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.gca()
# y_hat_all_short.groupby(y_hat_all_short.index.date).get_group(date_).iloc[:,0].plot(ax=ax,label='hat')
# y_true_all.groupby(y_hat_all_short.index.date).get_group(date_).iloc[:,0].plot(ax=ax,label='true')
# ax.legend()
# plt.show()