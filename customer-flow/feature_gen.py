# coding: utf-8

import pandas as pd
import numpy as np

from itertools import groupby

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge



def add_penultimate_week(df_shop, target='pays_count'):
    shift_1 = df_shop[target].shift(7)
    shift_2 = df_shop[target].shift(14)
    
    second_in_biweek = df_shop.week_id % 2

    df_shop['penultimate'] = 0.0
    df_shop.loc[second_in_biweek == 0, 'penultimate'] = shift_1[second_in_biweek == 0]
    df_shop.loc[second_in_biweek == 1, 'penultimate'] = shift_2[second_in_biweek == 1]

    df_shop['penultimate_null'] = df_shop.penultimate.isnull()
    df_shop['pays_two_weeks_ago'] = shift_2 
    df_shop['second_in_biweek'] = second_in_biweek


# out of operations 

def shift_index(series):
    s = series.copy()
    s.index = s.index - 1
    return s

def add_out_of_operation_features(df_shop):
    ooo_sum = df_shop.groupby('biweek_id').out_of_operation.sum()
    ooo_cnt = df_shop.groupby('biweek_id').out_of_operation.count()

    ooo_sum_1 = shift_index(ooo_sum)
    ooo_sum_2 = shift_index(ooo_sum_1)
    ooo_sum_3 = shift_index(ooo_sum_2)
    ooo_sum_4 = shift_index(ooo_sum_3)

    ooo_cnt_1 = shift_index(ooo_cnt)
    ooo_cnt_2 = shift_index(ooo_cnt_1)
    ooo_cnt_3 = shift_index(ooo_cnt_2)
    ooo_cnt_4 = shift_index(ooo_cnt_3)

    ooo_mean_1 = ooo_sum_1 / ooo_cnt_1
    ooo_mean_2 = (ooo_sum_1 + ooo_sum_2) / \
                 (ooo_cnt_1 + ooo_cnt_2)
    ooo_mean_3 = (ooo_sum_1 + ooo_sum_2 + ooo_sum_3) / \
                 (ooo_cnt_1 + ooo_cnt_2 + ooo_cnt_3)
    ooo_mean_4 = (ooo_sum_1 + ooo_sum_2 + ooo_sum_3 + ooo_sum_4) / \
                 (ooo_cnt_1 + ooo_cnt_2 + ooo_cnt_3 + ooo_cnt_4)

    df_shop['ooo_mean_1'] = ooo_mean_1[df_shop.biweek_id].reset_index(drop=1)
    df_shop['ooo_mean_2'] = ooo_mean_2[df_shop.biweek_id].reset_index(drop=1)
    df_shop['ooo_mean_3'] = ooo_mean_3[df_shop.biweek_id].reset_index(drop=1)
    df_shop['ooo_mean_4'] = ooo_mean_4[df_shop.biweek_id].reset_index(drop=1)


# weekly trend features

def add_weekly_overall_trends(df_shop, regressor, trend_name, coeff_name, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    df_shop[trend_name] = np.nan
    df_shop[coeff_name] = np.nan

    for m in range(biweek_max - 1, 0, -1):
        train_idx = df_shop.biweek_id >= m
        test_idx = df_shop.biweek_id == (m - 1)

        df_train = df_shop[train_idx]

        y = df_train[target]
        not_null = ~y.isnull()
        if not_null.sum() < 7:
            continue

        x = -df_train[regressor]
        x_not_null = x[not_null].values.reshape(-1, 1)
        y = y[not_null].values
        lr = Ridge(alpha=1).fit(x_not_null, y)

        if m == biweek_max - 1:
            x = x.values.reshape(-1, 1)
            df_shop.loc[train_idx, trend_name] = lr.predict(x)
            df_shop.loc[train_idx, coeff_name] = lr.coef_[0]

        df_test = df_shop[test_idx]
        x = -df_test[regressor].values.reshape(-1, 1)

        df_shop.loc[test_idx, trend_name] = lr.predict(x)
        df_shop.loc[test_idx, coeff_name] = lr.coef_[0]


def add_weekly_dow_trends(df_shop, regressor, trend_name, coeff_name, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    df_shop[trend_name] = np.nan
    df_shop[coeff_name] = np.nan

    for m in range(biweek_max - 1, 0, -1):
        train_idx = df_shop.biweek_id >= m
        test_idx = df_shop.biweek_id == (m - 1)

        df_train = df_shop[train_idx]
        df_test = df_shop[test_idx]

        for i in range(7):
            dow_idx = (df_train.dow == i)
            dow_idx_test = (df_test.dow == i)

            y = df_train[dow_idx][target]
            not_null = ~y.isnull()
            if not_null.sum() < 4:
                continue

            x = -df_train[regressor][dow_idx]
            x_not_null = x[not_null].values.reshape(-1, 1)
            y = y[not_null].values
            lr = Ridge(alpha=1).fit(x_not_null, y)

            if m == biweek_max - 1:
                x = x.values.reshape(-1, 1)
                df_shop.loc[train_idx & dow_idx, trend_name] = lr.predict(x)
                df_shop.loc[train_idx & dow_idx, coeff_name] = lr.coef_[0]

            x = -df_test[regressor][dow_idx_test].values.reshape(-1, 1)

            df_shop.loc[test_idx & dow_idx_test, trend_name] = lr.predict(x)
            df_shop.loc[test_idx & dow_idx_test, coeff_name] = lr.coef_[0]

def add_weekly_weekend_trends(df_shop, regressor, trend_name, coeff_name, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    df_shop[trend_name] = np.nan
    df_shop[coeff_name] = np.nan

    for m in range(biweek_max - 1, 0, -1):
        train_idx = df_shop.biweek_id >= m
        test_idx = df_shop.biweek_id == (m - 1)

        df_train = df_shop[train_idx]
        df_test = df_shop[test_idx]

        for i in [True, False]:
            dow_idx = (df_train.is_weekend == i)
            dow_idx_test = (df_test.is_weekend == i)

            y = df_train[dow_idx][target]
            not_null = ~y.isnull()
            if not_null.sum() < 4:
                continue

            x = -df_train[regressor][dow_idx]
            x_not_null = x[not_null].values.reshape(-1, 1)
            y = y[not_null].values
            lr = Ridge(alpha=1).fit(x_not_null, y)

            if m == biweek_max - 1:
                x = x.values.reshape(-1, 1)
                df_shop.loc[train_idx & dow_idx, trend_name] = lr.predict(x)
                df_shop.loc[train_idx & dow_idx, coeff_name] = lr.coef_[0]

            x = -df_test[regressor][dow_idx_test].values.reshape(-1, 1)

            df_shop.loc[test_idx & dow_idx_test, trend_name] = lr.predict(x)
            df_shop.loc[test_idx & dow_idx_test, coeff_name] = lr.coef_[0]


def add_weekly_trends_features(df_shop, target='pays_count'):
    add_weekly_overall_trends(df_shop, 'week_id', 'weekly_trend', 'weekly_coef', target)
    add_weekly_overall_trends(df_shop, 'biweek_id', 'biweekly_trend', 'biweekly_coef', target)
    add_weekly_dow_trends(df_shop, 'biweek_id', 'biweekly_dow_trend', 'biweekly_dow_coef', target)
    add_weekly_weekend_trends(df_shop, 'biweek_id', 'biweekly_weekend_trend', 'biweekly_weekend_coef', target)


# daily trends - overall, day of week, weekend/not

def add_overall_trend_feature(df_shop, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    trend_name = 'trend_overall'
    coeff_name = 'trend_overall_coeff'
    df_shop[trend_name] = np.nan
    df_shop[coeff_name] = np.nan

    for m in range(biweek_max - 1, 0, -1):
        train_idx = df_shop.biweek_id >= m
        test_idx = df_shop.biweek_id == (m - 1)

        df_train = df_shop[train_idx]

        y = df_train[target]
        not_null = ~y.isnull()
        if not_null.sum() <= 7:
            continue

        x = df_train.days_from_beginning
        x_not_null = x[not_null].values.reshape(-1, 1)
        y = y[not_null].values
        lr = Ridge(alpha=1).fit(x_not_null, y)

        if m == biweek_max - 1:
            x = x.values.reshape(-1, 1)
            df_shop.loc[train_idx, trend_name] = lr.predict(x)
            df_shop.loc[train_idx, coeff_name] = lr.coef_[0]

        df_test = df_shop[test_idx]
        x = df_test.days_from_beginning.values.reshape(-1, 1)

        df_shop.loc[test_idx, trend_name] = lr.predict(x)
        df_shop.loc[test_idx, coeff_name] = lr.coef_[0]


def add_dow_trend_feature(df_shop, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    trend_name = 'trend_overall_dow'
    coeff_name = 'trend_overall_dow_coeff'
    df_shop[trend_name] = np.nan
    df_shop[coeff_name] = np.nan

    for m in range(biweek_max - 1, 0, -1):
        train_idx = df_shop.biweek_id >= m
        test_idx = df_shop.biweek_id == (m - 1)

        df_train = df_shop[train_idx]
        df_test = df_shop[test_idx]

        for i in range(7):
            dow_idx = (df_train.dow == i)
            dow_idx_test = (df_test.dow == i)

            y = df_train[dow_idx][target]
            not_null = ~y.isnull()
            if not_null.sum() < 5:
                continue

            x = df_train[dow_idx].days_from_beginning
            x_not_null = x[not_null].values.reshape(-1, 1)
            y = y[not_null].values
            lr = Ridge(alpha=1).fit(x_not_null, y)

            if m == biweek_max - 1:
                x = x.values.reshape(-1, 1)
                df_shop.loc[train_idx & dow_idx, trend_name] = lr.predict(x)
                df_shop.loc[train_idx & dow_idx, coeff_name] = lr.coef_[0]

            x = df_test[dow_idx_test].days_from_beginning
            x = x.values.reshape(-1, 1)

            df_shop.loc[test_idx & dow_idx_test, trend_name] = lr.predict(x)
            df_shop.loc[test_idx & dow_idx_test, coeff_name] = lr.coef_[0]


def add_weekend_trend_feature(df_shop, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    trend_name = 'trend_overall_weekend'
    coeff_name = 'trend_overall_weekend_coeff'
    df_shop[trend_name] = np.nan
    df_shop[coeff_name] = np.nan

    for m in range(biweek_max - 1, 0, -1):
        train_idx = df_shop.biweek_id >= m
        test_idx = df_shop.biweek_id == (m - 1)

        df_train = df_shop[train_idx]
        df_test = df_shop[test_idx]

        for i in [True, False]:
            dow_idx = (df_train.is_weekend == i)
            dow_idx_test = (df_test.is_weekend == i)

            y = df_train[dow_idx][target]
            not_null = ~y.isnull()
            if not_null.sum() <= 7:
                continue

            x = df_train[dow_idx].days_from_beginning
            x_not_null = x[not_null].values.reshape(-1, 1)
            y = y[not_null].values
            lr = Ridge(alpha=1).fit(x_not_null, y)

            if m == biweek_max - 1:
                x = x.values.reshape(-1, 1)
                df_shop.loc[train_idx & dow_idx, trend_name] = lr.predict(x)
                df_shop.loc[train_idx & dow_idx, coeff_name] = lr.coef_[0]

            x = df_test[dow_idx_test].days_from_beginning
            x = x.values.reshape(-1, 1)

            df_shop.loc[test_idx & dow_idx_test, trend_name] = lr.predict(x)
            df_shop.loc[test_idx & dow_idx_test, coeff_name] = lr.coef_[0]



def add_trend_features(df_shop, target='pays_count'):
    add_overall_trend_feature(df_shop, target)
    add_dow_trend_feature(df_shop, target)
    add_weekend_trend_feature(df_shop, target)



# trends in windows

def add_window_trend_overall_features(df_shop, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    for biweeks_past in [2, 3, 4, 5, 6, 12, 18]:
        trend_name = 'trend_%d' % biweeks_past
        trend_coef_name = 'trend_coef_%d' % biweeks_past
        df_shop[trend_name] = np.nan
        df_shop[trend_coef_name] = np.nan

        for m in range(biweek_max, biweeks_past, -1):
            m_past = m - biweeks_past
            train_idx = (df_shop.biweek_id >= m_past) & (df_shop.biweek_id <= m)
            test_idx = df_shop.biweek_id == (m_past - 1)

            df_rolling_train = df_shop[train_idx]
            df_rolling_test = df_shop[test_idx]

            y = df_rolling_train[target]
            not_null = ~y.isnull()
            if not_null.sum() <= 7:
                continue
        
            x = df_rolling_train.days_from_beginning
            x_not_null = x[not_null].values.reshape(-1, 1)
            y = y[not_null].values
            lr = Ridge(alpha=1).fit(x_not_null, y)

            if m == biweek_max:
                x = x.values.reshape(-1, 1)
                df_shop.loc[train_idx, trend_name] = lr.predict(x)
                df_shop.loc[train_idx, trend_coef_name] = lr.coef_[0]

            x_val = df_rolling_test.days_from_beginning.values.reshape(-1, 1)
            df_shop.loc[test_idx, trend_name] = lr.predict(x_val)
            df_shop.loc[test_idx, trend_coef_name] = lr.coef_[0]


def add_window_trend_dow_features(df_shop, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    for biweeks_past in [3, 4, 5, 6, 12, 18]:
        trend_name = 'trend_dow_%d' % biweeks_past
        trend_coef_name = 'trend_dow_coef_%d' % biweeks_past
        df_shop[trend_name] = np.nan
        df_shop[trend_coef_name] = np.nan

        for m in range(biweek_max, biweeks_past, -1):
            m_past = m - biweeks_past
            train_idx = (df_shop.biweek_id >= m_past) & (df_shop.biweek_id <= m)
            test_idx = df_shop.biweek_id == (m_past - 1)

            df_rolling_train = df_shop[train_idx]
            df_rolling_test = df_shop[test_idx]
            
            for i in range(7):
                dow_idx = (df_rolling_train.dow == i)
                dow_idx_test = (df_rolling_test.dow == i)

                y = df_rolling_train[dow_idx][target]
                not_null = ~y.isnull()
                if not_null.sum() <= 4:
                    continue

                x = df_rolling_train[dow_idx].days_from_beginning
                x_not_null = x[not_null].values.reshape(-1, 1)
                y = y[not_null].values
                lr = Ridge(alpha=1).fit(x_not_null, y)

                if m == biweek_max:
                    x = x.values.reshape(-1, 1)
                    df_shop.loc[train_idx & dow_idx, trend_name] = lr.predict(x)
                    df_shop.loc[train_idx & dow_idx, trend_coef_name] = lr.coef_[0]

                x_val = df_rolling_test[dow_idx_test].days_from_beginning.values.reshape(-1, 1)
                df_shop.loc[test_idx & dow_idx_test, trend_name] = lr.predict(x_val)
                df_shop.loc[test_idx & dow_idx_test, trend_coef_name] = lr.coef_[0]


def add_window_trend_weekend_features(df_shop, target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    for biweeks_past in [3, 4, 5, 6, 12, 18]:
        trend_name = 'trend_weekend_%d' % biweeks_past
        trend_coef_name = 'trend_weekend_coef_%d' % biweeks_past
        df_shop[trend_name] = np.nan
        df_shop[trend_coef_name] = np.nan

        for m in range(biweek_max, biweeks_past, -1):
            m_past = m - biweeks_past
            train_idx = (df_shop.biweek_id >= m_past) & (df_shop.biweek_id <= m)
            test_idx = df_shop.biweek_id == (m_past - 1)

            df_rolling_train = df_shop[train_idx]
            df_rolling_test = df_shop[test_idx]
            
            for i in [True, False]:
                dow_idx = (df_rolling_train.is_weekend == i)
                dow_idx_test = (df_rolling_test.is_weekend == i)

                y = df_rolling_train[dow_idx][target]
                not_null = ~y.isnull()
                if not_null.sum() <= 4:
                    continue

                x = df_rolling_train[dow_idx].days_from_beginning
                x_not_null = x[not_null].values.reshape(-1, 1)
                y = y[not_null].values
                lr = Ridge(alpha=1).fit(x_not_null, y)

                if m == biweek_max:
                    x = x.values.reshape(-1, 1)
                    df_shop.loc[train_idx & dow_idx, trend_name] = lr.predict(x)
                    df_shop.loc[train_idx & dow_idx, trend_coef_name] = lr.coef_[0]

                x_val = df_rolling_test[dow_idx_test].days_from_beginning.values.reshape(-1, 1)
                df_shop.loc[test_idx & dow_idx_test, trend_name] = lr.predict(x_val)
                df_shop.loc[test_idx & dow_idx_test, trend_coef_name] = lr.coef_[0]


def add_window_trend_features(df_shop, target='pays_count'):
    add_window_trend_overall_features(df_shop, target)
    add_window_trend_dow_features(df_shop, target)
    add_window_trend_weekend_features(df_shop, target)


# mean features in window

def add_window_mean_overall_features(df_shop, past_biweeks_list=[1, 2, 3, 4, 5, 6, 12], target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    for biweeks_past in past_biweeks_list:
        mean_name = 'mean_%d' % biweeks_past
        std_name = 'std_%d' % biweeks_past
        df_shop[mean_name] = np.nan
        df_shop[std_name] = np.nan

        for m in range(biweek_max, biweeks_past, -1):
            m_past = m - biweeks_past
            train_idx = (df_shop.biweek_id >= m_past) & (df_shop.biweek_id <= m)
            test_idx = df_shop.biweek_id == (m_past - 1)

            df_rolling_train = df_shop[train_idx]
            df_rolling_test = df_shop[test_idx]

            y = df_rolling_train[target]
            not_null = ~y.isnull()
            if not_null.sum() < 3:
                continue

            y = y[not_null].values
            mean = np.mean(y)
            std = np.std(y)

            if m == biweek_max:
                df_shop.loc[train_idx, mean_name] = mean
                df_shop.loc[train_idx, std_name] = std

            df_shop.loc[test_idx, mean_name] = mean
            df_shop.loc[test_idx, std_name] = std


def add_window_mean_dow_features(df_shop, past_biweeks_list=[2, 3, 4, 5, 6, 12], target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    for biweeks_past in past_biweeks_list:
        mean_name = 'dow_mean_%d' % biweeks_past
        std_name = 'dow_std_%d' % biweeks_past
        df_shop[mean_name] = np.nan
        df_shop[std_name] = np.nan

        for m in range(biweek_max, biweeks_past, -1):
            m_past = m - biweeks_past
            train_idx = (df_shop.biweek_id >= m_past) & (df_shop.biweek_id <= m)
            test_idx = df_shop.biweek_id == (m_past - 1)

            df_rolling_train = df_shop[train_idx]
            df_rolling_test = df_shop[test_idx]

            for i in range(7):
                dow_idx = (df_rolling_train.dow == i)
                dow_idx_test = (df_rolling_test.dow == i)

                y = df_rolling_train[target]
                not_null = ~y.isnull()
                if not_null.sum() <= 2:
                    continue

                y = y[not_null].values
                mean = np.mean(y)
                std = np.std(y)

                if m == biweek_max:
                    df_shop.loc[train_idx & dow_idx, mean_name] = mean
                    df_shop.loc[train_idx & dow_idx, std_name] = std

                df_shop.loc[test_idx & dow_idx_test, mean_name] = mean
                df_shop.loc[test_idx & dow_idx_test, std_name] = std


def add_window_mean_weekend_features(df_shop, past_biweeks_list=[2, 3, 4, 5, 6, 12], target='pays_count'):
    biweek_max = df_shop.biweek_id.max()

    for biweeks_past in past_biweeks_list:
        mean_name = 'weekend_mean_%d' % biweeks_past
        std_name = 'weekend_std_%d' % biweeks_past
        df_shop[mean_name] = np.nan
        df_shop[std_name] = np.nan

        for m in range(biweek_max, biweeks_past, -1):
            m_past = m - biweeks_past
            train_idx = (df_shop.biweek_id >= m_past) & (df_shop.biweek_id <= m)
            test_idx = df_shop.biweek_id == (m_past - 1)

            df_rolling_train = df_shop[train_idx]
            df_rolling_test = df_shop[test_idx]

            for i in [True, False]:
                dow_idx = (df_rolling_train.is_weekend == i)
                dow_idx_test = (df_rolling_test.is_weekend == i)

                y = df_rolling_train[target]
                not_null = ~y.isnull()
                if not_null.sum() <= 2:
                    continue

                y = y[not_null].values
                mean = np.mean(y)
                std = np.std(y)

                if m == biweek_max:
                    df_shop.loc[train_idx & dow_idx, mean_name] = mean
                    df_shop.loc[train_idx & dow_idx, std_name] = std

                df_shop.loc[test_idx & dow_idx_test, mean_name] = mean
                df_shop.loc[test_idx & dow_idx_test, std_name] = std


def add_window_mean_features(df_shop, target='pays_count'):
    add_window_mean_overall_features(df_shop, target)
    add_window_mean_dow_features(df_shop, target)
    add_window_mean_weekend_features(df_shop, target)




# last available features
    
def last_percentiles(df_shop, target='pays_count'):
    data = zip(df_shop[target], df_shop.biweek_id)

    prev_last = np.nan
    prev_p25 = np.nan
    prev_p75 = np.nan

    last_dict = {}
    p25_dict = {}
    p75_dict = {}

    for biweek, group in groupby(data, lambda x: x[1]):
        last_dict[biweek] = prev_last
        p25_dict[biweek] = prev_p25
        p75_dict[biweek] = prev_p75

        group = [p for (p, _) in group if not np.isnan(p)]
        if group:
            prev_last = group[-1]
            prev_p25 = np.percentile(group, 25)
            prev_p75 = np.percentile(group, 75)

    return last_dict, p25_dict, p75_dict


def add_last_features(df_shop, target='pays_count'):
    last_dict, p25_dict, p75_dict = last_percentiles(df_shop, target)

    df_shop['prev_biweek_last_value'] = df_shop.biweek_id.apply(last_dict.get)
    df_shop['prev_biweek_p25'] = df_shop.biweek_id.apply(p25_dict.get)
    df_shop['prev_biweek_p75'] = df_shop.biweek_id.apply(p75_dict.get)
    df_shop['prev_spread'] = df_shop['prev_biweek_p75'] - df_shop['prev_biweek_p25']
