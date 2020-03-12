#!/usr/bin/python
# -*- coding:utf-8 -*-

#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

func = lambda I, a1, a2, a3:(a1 + a2 * I + a3 * np.log(I)) * I

def forecast_(E, P_fit, power_, set_, ratio_test, K_Tc, start_time_k):
    RMSE_ = []

    lb = [-3.022665909989761, -0.0032645802674822265, -0.21590802965658423]
    ub = [2.721796172973509, -0.000342, 0.8838005111891243]

    y_hat_all = pd.DataFrame(columns=['var1(t)', 'var1(t+1)', 'var1(t+2)', 'var1(t+3)', 'var1(t+4)',
                                      'var1(t+5)', 'var1(t+6)', 'var1(t+7)', 'var1(t+8)', 'var1(t+9)',
                                      'var1(t+10)', 'var1(t+11)', 'var1(t+12)', 'var1(t+13)', 'var1(t+14)',
                                      'var1(t+15)'])
    y_true_all = pd.DataFrame(columns=['var1(t)', 'var1(t+1)', 'var1(t+2)', 'var1(t+3)', 'var1(t+4)',
                                       'var1(t+5)', 'var1(t+6)', 'var1(t+7)', 'var1(t+8)', 'var1(t+9)',
                                       'var1(t+10)', 'var1(t+11)', 'var1(t+12)', 'var1(t+13)', 'var1(t+14)',
                                       'var1(t+15)'])

    E = pd.DataFrame({'E': E[:len(power_)].values.reshape(-1)}, index=power_.index)

    for date in set_:

        if (date > set_[0]):
            # use set_2019[set_2019<date][-1] fit a new parameter
            clear_date = set_[set_ < date][-1]
            frame_to_fit = power_[clear_date * 96:(clear_date + 1) * 96]
            X_fit_1 = E[clear_date * 96:(clear_date + 1) * 96].copy()
            ratio_fit_1 = ratio_test[clear_date * 96:(clear_date + 1) * 96]
            K_Tc_1 = K_Tc[clear_date * 96:(clear_date + 1) * 96]
            Y_fit_1 = pd.DataFrame(
                {'Y_fit': frame_to_fit.values.reshape(-1) / (ratio_fit_1 * K_Tc_1)}, index=X_fit_1.index)

            X_fit_1 = X_fit_1[X_fit_1['E'] > 0]
            Y_fit_1 = Y_fit_1.loc[X_fit_1.index]

            popt_iter1, pcov_iter1 = curve_fit(func, X_fit_1.values.reshape(-1), Y_fit_1.values.reshape(-1),
                                               bounds=(lb, ub))

            P_I_iter_1 = np.array([0 if E.values.reshape(-1)[i] == 0 else func(E.values.reshape(-1)[i], *popt_iter1)
                                   for i in range(len(E))])
            P_I_iter_1[P_I_iter_1 < 0] = 0

            P_fit_iter_1 = P_I_iter_1 * ratio_test[:len(power_)] * K_Tc[:len(power_)]
            P_fit_iter_1 = pd.DataFrame(
                {'P': P_fit_iter_1[:len(power_)]}, index=power_.index)


        #         frame_P_fit_iter_1 = P_fit_iter_1[date*96:(date+1)*96]
        #             frame_P_fit_iter_1_ = series_to_supervised(frame_P_fit_iter_1, 0, 16)
        else:

            P_fit_iter_1 = pd.DataFrame({'P': P_fit[:len(power_)].values.reshape(-1)}, index=power_.index)

        frame_E = E[date * 96:(date + 1) * 96]
        frame_fit = P_fit_iter_1[date * 96:(date + 1) * 96]
        frame = power_[date * 96:(date + 1) * 96]
        frame.index = frame_fit.index

        frame_ = series_to_supervised(frame, 0, 16)
        frame_fit_ = series_to_supervised(frame_fit, 0, 16)

        row_sum = []
        for index, row in frame_fit_.iterrows():
            row_sum = row_sum + [row.sum()]
        for i in range(len(row_sum)):
            if row_sum[-i] != 0:
                start_num = -i
                start_index = frame_fit_.iloc[-i:-(i - 1), :].index
            if row_sum[i] != 0:
                end_num = i
                end_index = frame_fit_.iloc[i:(i + 1), :].index

        y_hat = frame_fit_.iloc[start_num:end_num + 1].copy()

        lb = [-3.022665909989761, -0.0032645802674822265, -0.21590802965658423]
        ub = [2.721796172973509, -0.000342, 0.8838005111891243]
        for i in range(start_time_k, len(y_hat) - 1):
            time = y_hat.iloc[i:i + 1].index[0]
            time_last = y_hat.iloc[i - 1:i].index[0]  # 上一时刻
            time_sunrise = frame_fit[frame_fit['P'] > 0].index[0]

            X_fit = frame_E.loc[time_sunrise:time]

            ratio_fit = pd.DataFrame({'ratio': ratio_test[:len(power_)]}, index=power_.index).loc[time_sunrise:time]
            K_Tc_fit = pd.DataFrame({'K_Tc': K_Tc[:len(power_)]}, index=power_.index).loc[time_sunrise:time]
            Y_fit = frame.loc[time_sunrise:time].values.reshape(-1
                                                                ) / (ratio_fit.values.reshape(-1
                                                                                              ) * K_Tc_fit.values.reshape(-1))

            popt_iter, pcov_iter = curve_fit(func, X_fit.values.reshape(-1), Y_fit,
                                             bounds=(lb, ub))

            P_I_iter = np.array([0 if E.values.reshape(-1)[i] == 0 else func(E.values.reshape(-1)[i], *popt_iter)
                                 for i in range(len(E))])
            P_I_iter[P_I_iter < 0] = 0

            P_fit_iter = P_I_iter * ratio_test[:len(power_)] * K_Tc[:len(power_)]
            P_fit_iter = pd.DataFrame({'P_fit': P_fit_iter[:len(power_)]}, index=power_.index)

            frame_P_fit_iter = P_fit_iter[date * 96:(date + 1) * 96]
            frame_P_fit_iter_ = series_to_supervised(frame_P_fit_iter, 0, 16)
            y_hat.loc[time] = frame_P_fit_iter_.loc[time]

        y_true = frame_.loc[y_hat.index]
        y_hat_all = y_hat_all.append(y_hat)
        y_true_all = y_true_all.append(y_true)

        rmse = np.sqrt(((y_hat - y_true) ** 2).mean())
        RMSE_.append(rmse)
    #         print(date, 'finished')

    return y_hat_all, y_true_all, RMSE_

