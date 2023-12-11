import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import tensorflow as tf

df = pd.read_csv('data/marketing/clean.csv')

# Training
x = df.drop(columns = 'deposit_bool')
y = df[['deposit_bool']]

scaler = pp.RobustScaler()
scaler.fit(x.values)
x_original = x
x = scaler.transform(x.values)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.compile(loss="binary_crossentropy",metrics=['accuracy'])
model.summary()

history = model.fit(x, y, epochs=50, batch_size=10, validation_split=0.2)
eval = model.evaluate(x, y)
print(eval)
predict = model.predict(x)
print(predict)

model.save("data/marketing/bank_mlp.h5")



