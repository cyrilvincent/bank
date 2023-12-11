import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import random
import sklearn.neural_network as nn
import pickle
import matplotlib.pyplot as plt

random.seed(0)

df = pd.read_csv('data/churn/churn_clean.csv', delimiter=',')
y = df.Exited
x = df.drop(["Exited"], axis=1)

scaler = pp.RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

model = nn.MLPClassifier(hidden_layer_sizes=(15,15,15,15,15,15))
model.fit(x, y)
print(f"Score: {model.score(x, y)}")



