import numpy as np
from sklearn.linear_model import Ridge

def remove_outliers(group, target='pays_count'):
    y = group[target]
    not_null = ~y.isnull()

    x = group.days_from_beginning
    x_not_null = x[not_null].values.reshape(-1, 1)
    y = y[not_null].values
    lr = Ridge(alpha=1).fit(x_not_null, y)

    trend_fit = lr.predict(x.values.reshape(-1, 1))

    mae = (group.pays_count - trend_fit).abs()
    std = mae.std()
    outliers = mae > (std * 4)
    
    group.loc[outliers, target] = np.nan