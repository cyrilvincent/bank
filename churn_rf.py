import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import random
import sklearn.ensemble as rf
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

model = rf.RandomForestClassifier()
model.fit(xtrain, ytrain)
print(f"Score: {model.score(xtest, ytest)}")
with open("data/churn/churn_rf.pickle", "wb") as f:
    pickle.dump((scaler, model), f)

ypred = model.predict(xtest)
print(classification_report(ytest,  ypred))

print(model.feature_importances_)
plt.bar(df.columns[1:], model.feature_importances_)
plt.xticks(rotation=45)
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(model.estimators_[0],
                 out_file='data/churn/churn.dot',
                 feature_names = df.columns[1:],
                 class_names = ["0", "1"],
                 rounded = True, proportion = False,
                 precision = 2, filled = True)

