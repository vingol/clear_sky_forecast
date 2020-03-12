#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
from clear_date_index import generate_clear_index
from scipy.optimize import curve_fit


def func_altitude(x, alpha, belta, c):
    return c + alpha * np.sin(2 * x * np.pi / 365) + \
        belta * np.cos(2 * x * np.pi / 365)


def func(I, a1, a2, a3):
    return (a1 + a2 * I + a3 * np.log(I)) * I


def max_oneday(df, day):
    return df.iloc[day * 96:(day + 1) * 96].max().values.tolist()[0]


def RMSE_oneday(y, y_hat):
    num = 96 - sum(y == 0)
    return np.sqrt(np.sum((y - y_hat) ** 2) / num)


def return_func(E, popt_test):
    return np.array([0 if E.values.reshape(-1)[i] ==
                     0 else func(E.values.reshape(-1)[i], *popt_test) for i in range(len(E))])


def fit_model_new(power, E, nwp, clear_set):
    '''
    :param power: 输入一年的功率，得到这一年功率拟合得到的晴空功率
    :param E: 计算得到的辐照度
    :param nwp: nwp，按照吉林的nwp的时间设置，开始时间为早上八点，所以程序中有一段把时间切换了，如果是正常的NWP可以把这段删掉
    :param clear_set: power计算出的晴空序列，
    :return:
    '''
    # 拟合温度， 除去温度对效率的影响， 得到的power_是除以温度系数的变量

    P_final = power.copy()

    P_max = power.max().values.tolist()[0]

    # index_of_E = pd.date_range(start='2017-1-01', end='2017-12-31 23:45:00', periods=35040)
    E.index = power.index
    E_max = E.max()

    T_amb1 = nwp.temperature[96 * 365 - 33:96 * 365].copy()
    T_amb2 = nwp.temperature[:96 * 365 - 33].copy()
    T_amb = T_amb1.append(T_amb2)
    T_amb = pd.DataFrame({'T_amb': T_amb.values.reshape(-1)},
                         index=power.index) - 274.15

    NOCT = 45.5  # from reference
    T_c = T_amb.values.reshape(-1) + E.values.reshape(-1) * (NOCT - 20) / 800
    K_Tc = 1 - 0.005 * (T_c[:365 * 96] - 25)

    power_ = pd.DataFrame(
        {'power': power[:365 * 96].values.reshape(-1) / K_Tc}, index=power.index)

    # 用于拟合的数据是晴天的数据，
    ratio_train = []
    for i in clear_set:
        E_k = E[i * 96:(i + 1) * 96]
        E_k_m = E_k.max().values
        P_k = power_[i * 96:(i + 1) * 96]
        P_k_m = P_k.max().values

        ratio_train = ratio_train + [P_k_m / E_k_m]

    X = np.array(clear_set)
    Y = np.array(ratio_train).reshape(-1)

    popt, pcov = curve_fit(func_altitude, X, Y)

    y_hat = func_altitude(X, *popt)

    ratio_test = func_altitude(np.arange(365), *popt)
    ratio_test = np.repeat(ratio_test, 96)

    index_2018 = power.index
    P_altitude = power_.values.reshape(-1) / ratio_test

    P_altitude = pd.DataFrame({'P_altitude': P_altitude}, index=index_2018)

    names = locals()
    parameter = {}
    for date in clear_set:
        # 拟合形状
        X2 = E.iloc[date * 96:(date + 1) * 96]
        Y2 = P_altitude.iloc[date * 96:(date + 1) * 96]

        X2 = X2[X2 > 1].dropna()
        Y2 = Y2.loc[X2.index]
        popt_2, pcov_2 = curve_fit(
            func, X2.values.reshape(-1), Y2.values.reshape(-1))

        P_I = np.array([0 if E.values.reshape(-1)[i] ==
                        0 else func(E.values.reshape(-1)[i], *popt_2) for i in range(len(E))])
        P_I[P_I < 0] = 0

        P_fit = P_I * ratio_test * K_Tc
        P_fit = pd.DataFrame({'P_fit': P_fit}, index=index_2018)

        # 保存参数
        parameter['%d' % date] = popt_2
        # names['P_single_%d' % date] = P_fit

        P_final.iloc[(date + 1) * 96:] = P_fit.iloc[(date + 1) * 96:]

    df_parameter = pd.DataFrame(parameter).T
    df_parameter.columns = ['a1', 'a2', 'a3']

    return P_final, P_altitude, ratio_test, K_Tc, df_parameter


def fit_model(power, E, K_Tc, clear_set):
    '''
    :param power: dataframe
    :param E:
    :param nwp:
    :param set_2017:
    :return: P_final_single: the same index with power
             ratio and K_Tc for further fit
    '''
    # 拟合温度， 除去温度对效率的影响， 得到的power_是除以温度系数的变量
    P_max = power.max().values.tolist()[0]

    # index_of_E = pd.date_range(start='2017-1-01', end='2017-12-31 23:45:00', periods=35040)
    E.index = power.index
    E_max = E.max()

    power_ = pd.DataFrame(
        {'power': power[:365 * 96].values.reshape(-1) / K_Tc}, index=power.index)

    # 用于拟合的数据是晴天的数据，
    ratio_train = []
    for i in clear_set:
        E_k = E[i * 96:(i + 1) * 96]
        E_k_m = E_k.max().values
        P_k = power_[i * 96:(i + 1) * 96]
        P_k_m = P_k.max().values

        ratio_train = ratio_train + [P_k_m / E_k_m]

    X = np.array(clear_set)
    Y = np.array(ratio_train).reshape(-1)

    popt, pcov = curve_fit(func_altitude, X, Y)

    y_hat = func_altitude(X, *popt)

    ratio_test = func_altitude(np.arange(365), *popt)
    ratio_test = np.repeat(ratio_test, 96)

    P_altitude = power_.values.reshape(-1) / ratio_test

    P_altitude = pd.DataFrame({'P_altitude': P_altitude}, index=power.index)

    names = locals()
    parameter = {}
    for date in clear_set:
        # 拟合形状
        X2 = E.iloc[date * 96:(date + 1) * 96]
        Y2 = P_altitude.iloc[date * 96:(date + 1) * 96]

        X2 = X2[X2 > 1].dropna()
        Y2 = Y2.loc[X2.index]
        popt_2, pcov_2 = curve_fit(
            func, X2.values.reshape(-1), Y2.values.reshape(-1))

        P_I = np.array([0 if E.values.reshape(-1)[i] ==
                        0 else func(E.values.reshape(-1)[i], *popt_2) for i in range(len(E))])
        P_I[P_I < 0] = 0

        P_fit = P_I * ratio_test * K_Tc
        P_fit = pd.DataFrame({'P_fit': P_fit}, index=power.index)

        # 保存参数
        parameter['%d' % date] = popt_2
        names['P_single_%d' % date] = P_fit

    # 分段拟合得到最终结果
    P_final_single = pd.DataFrame(
        {'P': np.arange(365 * 96)}, index=power.index)

    Section = [0]
    for i in range(len(clear_set)):
        if i == 0 and int((clear_set[i - 1] - 365 + clear_set[i]) / 2) > 0:
            Section = Section + \
                [int((clear_set[i - 1] - 365 + clear_set[i]) / 2)]
        elif i == len(clear_set) - 1 and int((clear_set[i - 1] - 365 + clear_set[i]) / 2) < 0:
            Section = Section + \
                [int((clear_set[i - 1] - 365 + clear_set[i]) / 2)]
        elif i != 0:
            Section = Section + [int((clear_set[i - 1] + clear_set[i]) / 2)]

    Section = Section + [365]

    for (i, date) in enumerate(clear_set):
        P_final_single.iloc[Section[i] *
                            96:Section[i +
                                       1] *
                            96] = names['P_single_%d' %
                                        date].iloc[Section[i] *
                                                   96:Section[i +
                                                              1] *
                                                   96].values
    P_final_single[P_final_single['P'] < 0] = 0

    df_parameter = pd.DataFrame(parameter).T
    df_parameter.columns = ['a1', 'a2', 'a3']

    return P_final_single, df_parameter, ratio_test, K_Tc


if __name__ == "__main__":
    df = pd.read_csv('basic_info.xlsx', )
