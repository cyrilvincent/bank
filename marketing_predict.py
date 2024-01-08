import pickle
import tensorflow as tf

data = [[59,2343,5,1042,0,1,0,False,False,False,False,False,False,False,False,False,False,False,True,False,True,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False,False,False,True,1.0,0.0]]
with open("data/marketing/bank_xgb.pickle", "rb") as f:
    scaler, _ = pickle.load(f)
x = scaler.transform(data)

model = tf.keras.models.load_model("data/marketing/bank_mlp.h5")

predict = model.predict(x)[0][0]
print(predict > 0.5)





