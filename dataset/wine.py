from sklearn.datasets import load_wine
data = load_wine()
data.keys()

print(data['DESCR'])
data['feature_names']
data['target_names']
data['target']
data['data']