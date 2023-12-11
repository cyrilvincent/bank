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
import tensorflow as tf

# V1 V28 : PCA (Réduction de dimension + anonymisation (données sensibles) + regularisation)

df = pd.read_csv('data/credit_card/creditcard.csv')
print(df.shape)
print(df.info())

x = df.drop('Class', axis = 1)
y = df['Class']

tf.random.set_seed(0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation=tf.nn.relu, input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(20, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.compile(loss="binary_crossentropy",metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=2, batch_size=2, validation_split=0.2, class_weight={0:1, 1:284807*10/492})
eval = model.evaluate(x, y, batch_size=200)
print(eval)
predict = model.predict(x, batch_size=200)
predict = np.where(predict > 0.5, 1, 0)
print(predict)

print(classification_report(y,  predict))
# Je veux mesurer les vrai positifs car les données sont TRES "déséquilibrées" : 492 frauds out of 284,807 transactions
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00    284315
#            1       0.88      0.10      0.19       492
# C'est comme un test covid quand c'est positif c'est sur car j'ai peu de faux positif
# Par contre je ne suis pazs bon car bcp de faux négatif
# Solutions : + d'epochs & + de data

model.save("data/credit_card/keras.h5")
