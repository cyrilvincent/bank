import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import random
import sklearn.neural_network as nn
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

random.seed(0)

df = pd.read_csv('data/churn/churn_clean.csv', delimiter=',')
y = df.Exited
x = df.drop(["Exited"], axis=1)

scaler = pp.RobustScaler()
scaler.fit(x.values)
x = scaler.transform(x)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
print(xtrain.shape, xtest.shape)

model = nn.MLPClassifier(hidden_layer_sizes=(15,15,15,15))
model.fit(xtrain, ytrain)
print(f"Score: {model.score(xtest, ytest)}")

ypred = model.predict(xtest)
print(classification_report(ytest,  ypred))

with open("data/churn/churn_mlp.pickle", "wb") as f:
    pickle.dump((scaler, model), f)


