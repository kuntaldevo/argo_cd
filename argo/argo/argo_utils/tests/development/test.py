
import pandas as pd
import numpy as np

@profile
def my_func():
    l = [1] * 10**6
#     l_ = l * 2
    l = l * 2
    return l
#     X = pd.DataFrame({
#         'int_col': np.random.randint(0, 10, 1000000),
#         'ohe_cat_col': np.random.randint(0, 2, 1000000),
#         'float_col': np.random.uniform(0, 1, 1000000)
#     })
    
#     int_mask = (X - X.round()).sum() == 0
#     int_cols = list(X.columns[int_mask])
#     float_cols = list(X.columns[~int_mask])
#     ohe_cat_cols = []
#     for col in int_cols:
#         unique_values = X[col].unique()
#         unique_values.sort()
#         unique_values = set(unique_values)
#         if unique_values == {0, 1}:
#             ohe_cat_cols.append(col)
#     return int_cols, ohe_cat_cols, float_cols

if __name__ == '__main__':
    my_func()
