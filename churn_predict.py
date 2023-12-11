import pickle

with open("data/churn/churn_rf.pickle", "rb") as f:
    scaler, model = pickle.load(f)

x = [[619,42,2,0.0,1,101348.88,0.0,0.047619047619047616,14.738095238095237,1,1,0]]
x = scaler.transform(x)
y = model.predict(x)
print(y)
