#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from clear_sky_irradiance import clear_sky_model
from clear_date_index import generate_clear_index_new
from fit_model import fit_model
from forecast import series_to_supervised, forecast_

def main_cs():
    data_root = os.path.abspath("/Users/mayuan/Downloads/projects/data_jilin")

    # basic info
    df = pd.read_excel(r'basic_info.xlsx', sheet_name='solar')

    k = 2
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

    # calculate P_fit
    P_fit, df_parameter, ratio_test, _ = fit_model(power_2018, E, K_Tc_2018, clear_set)

    # forecast
    start_time_k = 20
    y_hat_all, y_true_all, RMSE_ = forecast_(
        E, P_fit, power_2018, clear_set, ratio_test, K_Tc_2018, 20)

    return y_hat_all, y_true_all, RMSE_