#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def func(I, a1, a2, a3):
    return (a1+a2*I+a3*np.log(I))*I
def fit_iter(P_fit, set_, power, E, ratio_test, K_Tc):

    lb = [-3.022665909989761, -0.0032645802674822265, -0.21590802965658423]
    ub = [2.721796172973509, -0.000342, 0.8838005111891243]

    if set == []:
        P_fit_iter = P_fit.copy()
    else:
        P_fit_iter = pd.DataFrame({'P': np.arange(len(power))}, index=power.index)
        P_fit_iter[:(set_[0]+1)*96] = P_fit[:(set_[0]+1)*96].values.reshape(-1).tolist()
        for clear_date in set_:

            frame_to_fit = power[clear_date*96:(clear_date+1)*96]
            X_fit_1 = E[clear_date*96:(clear_date+1)*96].copy()
            ratio_fit_1 = ratio_test[clear_date*96:(clear_date+1)*96]
            K_Tc_1 = K_Tc[clear_date*96:(clear_date+1)*96]
            Y_fit_1 = pd.DataFrame(
                {'Y_fit': frame_to_fit.values.reshape(-1)/(ratio_fit_1*K_Tc_1)}, index=X_fit_1.index)

            X_fit_1 = X_fit_1[X_fit_1['E'] > 0]
            Y_fit_1 = Y_fit_1.loc[X_fit_1.index]

            popt_iter1, pcov_iter1 = curve_fit(func, X_fit_1.values.reshape(-1), Y_fit_1.values.reshape(-1),
                                               bounds=(lb, ub))

            P_I_iter_1 = np.array([0 if E.values.reshape(-1)[i] == 0 else func(E.values.reshape(-1)[i], *popt_iter1)
                                   for i in range(len(E))])
            P_I_iter_1[P_I_iter_1 < 0] = 0

            # print(len(power))
            # print(len(P_I_iter_1[:len(power)]))
            # print(len(ratio_test[:len(power)]))
            # print(len(K_Tc))

            P_fit_iter_1 = P_I_iter_1[:len(power)]*ratio_test[:len(power)]*K_Tc
            P_fit_iter_1 = pd.DataFrame(
                {'P': P_fit_iter_1}, index=power.index)

            P_fit_iter.iloc[(clear_date+1)*96:len(power)] = P_fit_iter_1.iloc[(clear_date+1)*96:]

    return pd.DataFrame(P_fit_iter[:len(power)])