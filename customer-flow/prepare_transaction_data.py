
import pandas as pd
import numpy as np

import feather

from tqdm import tqdm



df_user_pay = pd.read_csv('data/user_pay.txt', header=None)
df_user_pay.columns = ['user_id', 'shop_id', 'ts']
df_user_pay.shop_id = df_user_pay.shop_id.astype('uint16')
df_user_pay.user_id = df_user_pay.user_id.astype('uint32')
df_user_pay.ts = pd.to_datetime(df_user_pay.ts)
df_user_pay['day'] = df_user_pay.ts.dt.floor(freq='d')


df_pays = df_user_pay.groupby(by=['shop_id', 'day']).user_id.count().reset_index()
df_pays.rename(columns={'user_id': 'pays_count'}, inplace=1)

df_pays.shop_id = df_pays.shop_id.astype('uint16')
df_pays.pays_count = df_pays.pays_count.astype('uint16')


def fix_holes(df, day_min=None, day_max=None):
    if day_min is None:
        day_min = df.day.min()
    if day_max is None:
        day_max = df.day.max()

    td = day_max - day_min
    if len(df) == td.days + 1:
        return df

    idx = pd.date_range(day_min, day_max)

    df = df.reset_index(drop=1)
    df = df.set_index('day').reindex(index=idx).reset_index()
    df.rename(columns={'index': 'day'}, inplace=1)

    return df



shops = df_pays.shop_id.unique()
shops = sorted(shops)


max_day = pd.to_datetime('2016-10-31')
holes_fixed = []

for i in tqdm(shops):
    df = df_pays[df_pays.shop_id == i].reset_index(drop=1)
    del df['shop_id']
    df = fix_holes(df, day_max=max_day)
    df.insert(0, 'shop_id', i)

    first_day = df.day.min()
    days_from_beginning = (df.day - first_day).apply(lambda x: x.days)
    df.insert(2, 'days_from_beginning', days_from_beginning)

    holes_fixed.append(df)


df_pays = pd.concat(holes_fixed)


last_day = df_pays.day.max()
dt = (df_pays.day - last_day).apply(lambda d: (d.days - 1) // 7)

df_pays['week_id'] = -dt
df_pays['biweek_id'] = (df_pays.week_id + 1) // 2



def generate_test_df(df_shop):
    df_shop_test = pd.DataFrame()

    shop_id = df_shop.shop_id.iloc[0]
    df_shop_test['shop_id'] = [shop_id] * 14

    df_shop_test['day'] = pd.date_range('2016-11-01', '2016-11-14')

    days_max = df_shop.days_from_beginning.max()
    df_shop_test['days_from_beginning'] = np.arange(days_max + 1, days_max + 15)
    df_shop_test['pays_count'] = np.nan
    df_shop_test['week_id'] = [0] * 7 + [-1] * 7
    df_shop_test['biweek_id'] = 0

    return df_shop_test


dfs = []

for i in tqdm(shops):
    df_shop = df_pays[df_pays.shop_id == i]
    dfs.append(df_shop)
    df_shop_test = generate_test_df(df_shop)
    dfs.append(df_shop_test)


df_pays = pd.concat(dfs).reset_index(drop=1)

df_pays['dow'] = df_pays.day.dt.dayofweek.astype('uint8')
df_pays['is_weekend'] = df_pays.dow.isin([5, 6])

feather.write_dataframe(df_pays, 'data/df_pays_na_test.feather')