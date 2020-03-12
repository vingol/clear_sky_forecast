#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def minus_pow(x,y):
    return np.sign(x) * (np.abs(x)) ** (y)

def clear_sky_model(latitude, lontitude):
    n = np.arange(1, 366)
    n = np.repeat(n, 96)

    # 太阳赤纬角
    delta = 23.45*np.sin((360*(n+284)/365)*np.pi/180)
    delta = delta*np.pi/180
    # 37.550001	106.583332
    # 真太阳时
    c = np.arange(0,24,0.25)
    LST = np.tile(c,365)
    B = 360*(n-81)/364
    B = B*np.pi/180
    ET = 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B) #校正因子，需要重复96次(一天96个点)
    l_st = 120
    l_local = lontitude

    ST = LST + (ET -4*(l_st-l_local))/60
    w = 15*(ST-12)
    w = w*np.pi/180

    # 太阳高度角
    fai = latitude*np.pi/180
    sin_alpha = np.sin(fai)*np.sin(delta) + np.cos(fai)*np.cos(delta)*np.cos(w)
    alpha = np.arcsin(sin_alpha)*180/np.pi

    # 大气光学质量
    m = 1/(sin_alpha + 0.50572*(np.sign(6.07995+alpha) * (np.abs(6.07995+alpha)) ** (-1.6364)))
    m = pd.DataFrame({'m':m})
    index = m[m['m']<=0].index
    # m.iloc[index] = 0
    m = m.values
    m = m.reshape(-1)
    # m = [m.iloc[i] if m.iloc[i]>0
    #      else 0
    #      for i in range(365*96)]

    t_b_ = np.array([0.254, 0.285, 0.361, 0.401, 0.461, 0.545, 0.499, 0.440, 0.423, 0.401, 0.326, 0.261])
    t_d_ = np.array([2.415, 2.239, 1.997, 2.002, 1.875, 1.729, 1.925, 2.109, 2.035, 1.991, 2.186, 2.447])

    dom = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    t_b = []
    t_d = []
    for i in range(12):
        t_b = t_b+ [t_b_[i]]*dom[i]
        t_d = t_d+ [t_d_[i]]*dom[i]

    t_b = np.array(t_b).reshape(-1)
    t_d = np.array(t_d).reshape(-1)

    t_b = np.repeat(t_b, 96)
    t_d = np.repeat(t_d, 96)

    b = 1.219 - 0.043*t_b - 0.151*t_d - 0.204*t_b*t_d
    d = 0.202 + 0.852*t_b - 0.007*t_d - 0.357*t_b*t_d

    # 直接辐射和散射辐射的计算
    E_sc = 1366.1
    m_b = []
    m_d = []
    for i in range(365*96):
        # m_b = m_b + [m[i]**b[i]]
        # m_d = m_d + [m[i]**d[i]]
        m_b = m_b + [minus_pow(m[i], b[i])]
        m_d = m_d + [minus_pow(m[i], d[i])]
    m_b = np.array(m_b)
    m_d = np.array(m_d)

    E_b = E_sc*np.exp(-t_b*m_b)
    E_d = E_sc*np.exp(-t_d*m_d)

    E = E_b*sin_alpha + E_d

    test = pd.DataFrame({'sin_alpha':sin_alpha}).copy()
    index_alpha = test[test['sin_alpha']<0].index
    E[index_alpha] = 0


    # plt.plot(np.arange(96), E[:96])
    # plt.show()

    E_out = pd.DataFrame({'E':E})

    return E_out


if __name__ == '__main__':
    root = os.path.abspath('.')
    dir_ = os.path.join(root, 'irradiance_cal')

    df = pd.read_excel(r'basic_info.xlsx', sheet_name='solar')
    # for k in range(len(df)):
    #     latitude = df['Lat'][k]
    #     lontitude = df['Lon'][k]
    #     E_out = clear_sky_model(latitude,lontitude)
    #
    #     file_name = str(df['ID'][k]) + '.csv'
    #     savepath = os.path.join(dir_, file_name)
    #
    #     E_out.to_csv(savepath)
    #     print(str(df['ID'][k]), ' finished')
    k = 0

    latitude = df['Lat'][k]
    lontitude = df['Lon'][k]
    E_out = clear_sky_model(latitude, lontitude)

