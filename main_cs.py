#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from clear_sky_irradiance import clear_sky_model
from clear_date_index import generate_clear_index_new
from fit_model import fit_model
from fit_model_iter import fit_iter


data_root = os.path.abspath("/Users/mayuan/Downloads/projects/data_jilin")

# basic info
df = pd.read_excel(r'basic_info.xlsx', sheet_name='solar')
#
# 容量没有发生太大变化的电站编号
# index_no_cap_change = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 19]
index_no_cap_change = [16]
for k in index_no_cap_change:
    # try:

    print(str(df['ID'][k]), 'start')

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
    T_c = T_amb_2018.values.reshape(-1) + \
        E.values.reshape(-1) * (NOCT - 20) / 800
    K_Tc_2018 = 1 - 0.005 * (T_c[:365 * 96] - 25)

    power_2018 = power[96 * 365:]

    # calculate clear index
    cap = power.max().tolist()[0]
    clear_set, _ = generate_clear_index_new(power_2018, cap)

    # calculate P_fit
    P_fit, df_parameter, ratio_test, _ = fit_model(
        power_2018, E, K_Tc_2018, clear_set)

    # 调整日出日落时段的偏差
    P_c_index = pd.DataFrame({'index': [(power.iloc[96 * 365 + i].values / P_fit.iloc[i].values)[
        0] if P_fit.iloc[i].values != 0 else 1 for i in range(len(P_fit))]}, index=P_fit.index)

    P_c_index_ = P_c_index.copy()
    for group, frame in P_fit.groupby(P_fit.index.date):

        list_frame = frame.values.reshape(-1).tolist()
        for i in range(len(list_frame)):
            if list_frame[-i] != 0:
                rise_num = -i
                rise_time = frame.iloc[-i:-(i - 1), :].index[0]
            if list_frame[i] != 0:
                set_num = i
                set_time = frame.iloc[i:(i + 1), :].index[0]

        if rise_time < set_time:
            index_matter = pd.date_range(
                start=rise_time,
                periods=3,
                freq='15min').append(
                pd.date_range(
                    start=set_time,
                    periods=3,
                    freq='-15min'))
            for index_ in index_matter:
                if P_c_index_.loc[index_].values[0] > 1.2 or P_c_index_.loc[index_].values[0] < 0.8:
                    P_c_index_.loc[index_] = 1

    P_fit.to_csv('P_fit_2018/P_fit_%s.csv' % str(df['ID'][k]))
    P_c_index_.to_csv('clear_index_2018/clear_index_%s.csv' %
                      str(df['ID'][k]))
    # clear_index_2019

    # calculate 2019 cs power

    dir_2019 = os.path.join(data_root, 'data_2019_new')
    filepath_power_2019 = os.path.join(dir_2019, 'solar', station_name)
    filepath_nwp_2019 = os.path.join(dir_2019, 'nwp_2019', nwp_name)

    power_2019 = pd.read_csv(
        filepath_power_2019,
        index_col=0,
        parse_dates=True)
    power_2019.columns = ['power']
    nwp_2019 = pd.read_csv(
        filepath_nwp_2019,
        index_col=0,
        parse_dates=True)

    T_amb1 = nwp_2019.temperature[-33:].copy()
    T_amb2 = nwp_2019.temperature[:-33].copy()
    T_amb_2019 = T_amb1.append(T_amb2)
    T_amb_2019 = pd.DataFrame({'T_amb': T_amb_2019.values.reshape(-1)},
                              index=power_2019[:-1].index) - 274.15

    NOCT = 45.5  # from reference
    T_c_2019 = T_amb_2019.values.reshape(-1) + E[:len(
        T_amb_2019)].values.reshape(-1) * (NOCT - 20) / 800
    K_Tc_2019 = 1 - 0.005 * (T_c_2019 - 25)

    clear_set_2019, _ = generate_clear_index_new(power_2019[:-1], cap)

    P_fit_iter = fit_iter(P_fit, clear_set_2019,
                          power_2019[:-1], E, ratio_test, K_Tc_2019)
    # 调整日出日落时段的偏差
    P_c_index_iter = pd.DataFrame(
        {
            'index': [
                (power_2019.iloc[i].values /
                 P_fit_iter.iloc[i].values)[0] if P_fit_iter.iloc[i].values != 0 else 1 for i in range(
                    len(P_fit_iter))]},
        index=P_fit_iter.index)

    P_c_index_iter_ = P_c_index_iter.copy()
    for group, frame in P_fit_iter.groupby(P_fit_iter.index.date):

        list_frame = frame.values.reshape(-1).tolist()
        for i in range(len(list_frame)):
            if list_frame[-i] != 0:
                rise_num = -i
                rise_time = frame.iloc[-i:-(i - 1), :].index[0]
            if list_frame[i] != 0:
                set_num = i
                set_time = frame.iloc[i:(i + 1), :].index[0]

        if rise_time < set_time:
            index_matter = pd.date_range(
                start=rise_time,
                periods=3,
                freq='15min').append(
                pd.date_range(
                    start=set_time,
                    periods=3,
                    freq='-15min'))
            for index_ in index_matter:
                if P_c_index_iter_.loc[index_].values[0] > 1.2 or P_c_index_iter_.loc[index_].values[0] < 0.8:
                    P_c_index_iter_.loc[index_] = 1

    P_fit_iter.to_csv('P_fit_2019/P_fit_%s.csv' % str(df['ID'][k]))
    P_c_index_iter_.to_csv(
        'clear_index_2019/clear_index_%s.csv' % str(df['ID'][k]))

    print(str(df['ID'][k]), 'finished')

    # except Exception as e:
    #     pass
    # continue

import datetime
df_to_plot = power.groupby(power.index.date).get_group(datetime.date(2018,10,18)).copy()
df_to_plot['P_fit'] = P_fit.groupby(P_fit.index.date).get_group(datetime.date(2018,10,18)).values

df_to_plot.plot()
import matplotlib.pyplot as plt
plt.show()