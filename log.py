from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
import pickle
import pandas as pd
import sys

if len(sys.argv) < 2:
	sys.exit("Error: incorrect number of arguments supplied. Usage: python train.py example.csv")

filepath = sys.argv[1]

# load data from csv
try:
  dataframe = pd.read_csv(filepath)
except:
  sys.exit("Error: csv file not found.")

features = dataframe.columns
target = 'increasePercent'
features = features.drop([target])

# test and train split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataframe[features], dataframe[target],test_size=.3)

model = LogisticRegression(max_iter=200, class_weight='balanced')
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print(model.score(X_test, Y_test))
print(precision_score(Y_test, Y_pred))
print(recall_score(Y_test, Y_pred))

pickle.dump(model, open('model.sav', 'wb'))
