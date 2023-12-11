import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import random
import sklearn.linear_model as lm
import sklearn.neighbors as nn
import sklearn.svm as svm
import pickle
import matplotlib.pyplot as plt

random.seed(0)

df = pd.read_csv('data/churn/churn_clean.csv', delimiter=',')
y = df.Exited
x = df.drop(["Exited"], axis=1)

scaler = pp.RobustScaler()
scaler.fit(x.values)
x = scaler.transform(x)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2)
print(xtrain.shape, xtest.shape)

model = lm.LinearRegression()
model.fit(xtrain, ytrain)
print(f"Score: {model.score(xtest, ytest)}")

model = nn.KNeighborsClassifier()
model.fit(xtrain, ytrain)
print(f"Score: {model.score(xtest, ytest)}")
with open("data/churn/churn_knn.pickle", "wb") as f:
    pickle.dump((scaler, model), f)

model = svm.SVC()
model.fit(xtrain, ytrain)
print(f"Score: {model.score(xtest, ytest)}")






