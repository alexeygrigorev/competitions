
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import feather

from tqdm import tqdm


# In[2]:

from outliers import remove_outliers


# In[3]:

df_pays = feather.read_dataframe('data/df_pays_na_test.feather')


# In[4]:

shops = df_pays.shop_id.unique()
shops = sorted(shops)


# In[5]:

from fbprophet import Prophet


# In[7]:

def add_prophet_features(df_shop):
    df = df_shop[['day', 'pays_count']].rename(columns={'day': 'ds', 'pays_count': 'y'})

    results = []
    biweek_max = df_shop.biweek_id.max()

    for m in range(biweek_max - 1, 0, -1):
        train_idx = df_shop.biweek_id >= m
        df_train = df[train_idx]

        not_null = ~df_train.y.isnull()
        if not_null.sum() < 7:
            continue

        p = Prophet().fit(df_train)
        future = p.make_future_dataframe(14, include_history=False)
        pred = p.predict(future)
        results.append(pred)

    df_res = pd.concat(results)
    df_res.columns = ['prophet_%s' % c for c in pred.columns]

    df_res = df_shop.merge(df_res, how='left', left_on='day', right_on='prophet_ds')
    del df_res['prophet_t'], df_res['prophet_ds']
    
    df_res.drop_duplicates('days_from_beginning', keep='last', inplace=1)

    if len(df_res) != len(df_shop):
        raise Exception("size doesn't match")

    return df_res



# In[7]:

dfs = []

for i in tqdm(shops):
    df_shop = df_pays[df_pays.shop_id == i].reset_index(drop=1)
    remove_outliers(df_shop)

    df_shop = add_prophet_features(df_shop)
    dfs.append(df_shop)


# In[8]:

truncated_dfs = []

for df in dfs:
    df = df.iloc[7 * 2 * 3:]
    truncated_dfs.append(df)


df_features = pd.concat(truncated_dfs).reset_index(drop=1)



feather.write_dataframe(df_features, 'features/prophet_features_proper.feather')

