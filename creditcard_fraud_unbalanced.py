import pickle
import random
import numpy as np
import pandas as pd
import sweetviz
import collections
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn.model_selection as ms
import sklearn.ensemble as rf
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# V1 V28 : may be result of a PCA Dimensionality reduction to protect user identities and sensitive features(v1-v28)

df = pd.read_csv('data/credit_card/creditcard.csv')
print(df.shape)
print(df.info())

# my_report = sweetviz.analyze(df)
# my_report.show_html()

plt.figure()
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1 , cmap="YlGnBu")
plt.show()

x = df.drop('Class', axis = 1)
y = df['Class']

random.seed(0)
xtrain, xtest , ytrain , ytest = ms.train_test_split(x,y,train_size = 0.8)

model = rf.RandomForestClassifier(n_jobs=5)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
with open("data/credit_card/rf.pickle", "wb") as f:
    pickle.dump(model, f)

confusion_matrix = confusion_matrix(ytest, ypred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
plt.figure()
cm_display.plot()
plt.show()

print(classification_report(ytest,  ypred))
# Je veux mesurer les vrai positifs car les données sont TRES "déséquilibrées" : 492 frauds out of 284,807 transactions
# Ici 66
