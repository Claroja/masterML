from sklearn.datasets import load_iris
data = load_iris()
data.keys()
data['data']
data['target']
data['target_names']
data['feature_names']
data['DESCR']
print(data['DESCR'])

import pandas as pd

df=pd.DataFrame(data['data'])
df.columns=[data['feature_names']]

pd.plotting.scatter_matrix(df, alpha=0.7, figsize=(14,8), diagonal='kde')