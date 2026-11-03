import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load("iris_model.pkl")
data = pd.read_csv("Iris (1).csv")

data = data.drop("Id",axis=1)

x  = data.drop("Species", axis = 1)
y = data["Species"]
print(x.head())
pred = model.predict(x)

acc = accuracy_score(pred, y)
print(acc)
assert len (pred) == len(y)
assert acc >= 0.50
print ('all tests passed')