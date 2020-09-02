#!/usr/bin/python
# -*- coding:utf-8 -*-

#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import itertools
import numpy as np
import pandas as pd

import itertools


def generate_clear_index_new(power, cap):

    power.columns = ['power']
    clear_date = []
    clear_index = []
    m = 0
    requirement_label = {}
    for group, frame in power.groupby(power.index.date):
        '''
        pre_variables
        '''
        flag = 1
        label = []

        frame_plus = frame[frame['power'] > 0]
        frame_diff = frame.diff().fillna(0)
        frame_plus_diff = frame_plus.diff().fillna(0)
        middle = int(len(frame_plus) / 2)
        index_middle = frame.iloc[middle:middle + 1].index

        # requirement 1
        if frame.max().values <= 0.7 * cap:
            flag = 0
        label.append(flag)

        # requirement 2
        frame_plus_diff_down = frame_plus_diff[middle:].copy()
        down = frame_plus_diff_down[frame_plus_diff_down['power'] < 0].count().tolist()[
            0]

        frame_plus_diff_up = frame_plus_diff[:middle].copy()
        up = frame_plus_diff_up[frame_plus_diff_up['power'] > 0].count().values.tolist()[
            0]

        if up < 0.8 * middle or down < 0.8 * middle:
            flag = 0
        label.append(flag)

        # requirement 3
        judge_3 = frame_plus_diff.copy()
        judge = [
            1 if abs(
                frame_plus_diff.iloc[i].values) < 0.005 *
            cap else 0 for i in range(
                len(frame_plus_diff))]

        k_group = []
        v_list = []

        for k, v in itertools.groupby(judge):
            k_group.append(k)
            v_list.append(len(list(v)))

        for i in range(len(v_list)):
            if v_list[i] > 8 & k_group[i] == 1:
                flag = 0
        label.append(flag)

        # requirement 4
        judge_4 = [
            1 if abs(
                (frame_diff.iloc[i] /
                 cap).tolist()[0]) > 0.2 else 0 for i in range(
                len(frame))]
        if np.array(judge_4).sum() > 0.1 * len(frame_plus_diff):
            flag = 0
        label.append(flag)

        # requirement 5
        wavenum = 1  # 条件
        judge_5 = 0
        epison = 0.02 * cap
        for i in range(10, 70):
            if (frame.iloc[i] < frame.iloc[i - 1]).tolist()[0] & (frame.iloc[i] < frame.iloc[i + 1]).tolist()[0]:
                if (np.abs(frame_diff.iloc[i]) > epison).values.tolist()[0] | (
                        np.abs(frame_diff.iloc[i + 1]) > epison).values.tolist()[0]:
                    judge_5 = judge_5 + 1
                if (np.abs(frame_diff.iloc[i]) > 5 * epison).values.tolist()[0] | (
                        np.abs(frame_diff.iloc[i + 1]) > 5 * epison).values.tolist()[0]:
                    flag = 0
        if judge_5 >= wavenum:
            flag = 0
        label.append(flag)

        # requirement 6
        judge_6 = 0
        limit = 0.1
        symmetry_limit = 0.4
        for i in range(middle - 1):
            if np.abs((frame_plus.iloc[middle + i].values -
                       frame_plus.iloc[middle - i].values) / cap) > limit:
                judge_6 = judge_6 + 1
        if judge_6 > symmetry_limit * middle:
            flag = 0
        label.append(flag)

    #     final judge
    #     print(flag)
        if flag == 1:
            clear_date.append(group)
            clear_index.append(m)

        requirement_label[group] = label

        m = m + 1
    return clear_index, requirement_label


def generate_clear_index(power, cap):

    power.columns = ['power']
    clear_date = []
    clear_index = []
    m = 0
    requirement_label = {}

    for group, frame in power.groupby(power.index.date):
        '''
        pre_variables
        '''
        flag = 1
        label = []

        frame_plus = frame[frame['power'] > 0]
        frame_diff = frame.diff().fillna(0)
        frame_plus_diff = frame_plus.diff().fillna(0)
        middle = int(len(frame_plus) / 2)
        index_middle = frame.iloc[middle:middle + 1].index

        # requirement 1
        if frame.max().values <= 0.85 * cap:
            flag = 0
        label.append(flag)

        # requirement 2
        frame_plus_diff_down = frame_plus_diff[middle:].copy()
        down = frame_plus_diff_down[frame_plus_diff_down['power'] < 0].count().tolist()[
            0]

        frame_plus_diff_up = frame_plus_diff[:middle].copy()
        up = frame_plus_diff_up[frame_plus_diff_up['power'] > 0].count().values.tolist()[
            0]

        if up < 0.8 * middle or down < 0.8 * middle:
            flag = 0
        label.append(flag)

        # requirement 3
        judge_3 = frame_plus_diff.copy()
        judge = [
            1 if abs(
                frame_plus_diff.iloc[i].values) < 0.005 *
            cap else 0 for i in range(
                len(frame_plus_diff))]

        k_group = []
        v_list = []

        for k, v in itertools.groupby(judge):
            k_group.append(k)
            v_list.append(len(list(v)))

        for i in range(len(v_list)):
            if v_list[i] > 8 & k_group[i] == 1:
                flag = 0
        label.append(flag)

        # requirement 4
        judge_4 = [
            1 if abs(
                (frame_diff.iloc[i] /
                 cap).tolist()[0]) > 0.2 else 0 for i in range(
                len(frame))]
        if np.array(judge_4).sum() > 0.1 * len(frame_plus_diff):
            flag = 0
        label.append(flag)

        # requirement 5
        wavenum = 1  # 条件
        judge_5 = 0
        epison = 0.02 * cap
        for i in range(10, 70):
            if (frame.iloc[i] < frame.iloc[i -
                                           1]).tolist()[0] & (frame.iloc[i] < frame.iloc[i +
                                                                                         1]).tolist()[0]:
                if (np.abs(frame_diff.iloc[i]) > epison).values.tolist()[0] | (
                        np.abs(frame_diff.iloc[i + 1]) > epison).values.tolist()[0]:
                    judge_5 = judge_5 + 1
                if (np.abs(frame_diff.iloc[i]) > 5 * epison).values.tolist()[0] | (
                        np.abs(frame_diff.iloc[i + 1]) > 5 * epison).values.tolist()[0]:
                    flag = 0
        if judge_5 >= wavenum:
            flag = 0
        label.append(flag)

        # requirement 6
        judge_6 = 0
        limit = 0.05
        symmetry_limit = 0.4
        for i in range(middle - 1):
            if np.abs((frame_plus.iloc[middle + i].values -
                       frame_plus.iloc[middle - i].values) / cap) > limit:
                judge_6 = judge_6 + 1
        if judge_6 > symmetry_limit * middle:
            flag = 0
        label.append(flag)

        #     final judge
        #     print(flag)
        if flag == 1:
            clear_date.append(group)
            clear_index.append(m)

        requirement_label[group] = label

        m = m + 1
    return clear_index, requirement_label


if __name__ == '__main__':

    '''
    load data
    '''
    station_num = '503.csv'
    root = os.path.abspath('/Users/mayuan/Downloads/projects/data_jilin')
    dir_ = os.path.join(root, 'data', 'Power')
    filepath_power = os.path.join(dir_, '503.csv')
    # filepath_E = os.path.join(root, 'irradiance_cal', '503.csv')

    power = pd.read_csv(filepath_power, index_col=0, parse_dates=True)

    cap = power.max().tolist()[0]

    # E = pd.read_csv(filepath_E, index_col=0)
    # index_of_E = pd.date_range(
    #     start='2017-1-01',
    #     end='2017-12-31 23:45:00',
    #     periods=35040)
    # E.index = index_of_E

    # E_max = E.max().tolist()[0]
    clear_index = generate_clear_index_new(power, cap)
    clear_index_new = generate_clear_index_new(power, cap)

    dir_2019 = os.path.join(root, 'data_2019_new', 'solar')
    filepath_power_2019 = os.path.join(dir_2019, '503.csv')
    power_2019 = pd.read_csv(
        filepath_power_2019,
        index_col=0,
        parse_dates=True)
    power_2019.drop(index=power_2019.iloc[-1:].index, inplace=True)
    power_2019.columns = ['power']
    set_2019, label_2019 = generate_clear_index(power_2019, cap)
    set_2019_new, label_2019_new = generate_clear_index_new(power_2019, cap)
