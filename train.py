from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import sys

if len(sys.argv) < 2:
	sys.exit("Error: incorrect number of arguments supplied. Usage: python train.py example.csv optional_feature_1 optional_feature_2...")

filepath = sys.argv[1]

# load data from csv
try:
  dataframe = pd.read_csv(filepath)
except:
  sys.exit("Error: csv file not found.")

# default features
features = ['bookValue','marketCap','dividendYield','peRatio','pbRatio','currentRatio','quickRatio','stockPrice','eps']

# check for custom features
if len(sys.argv) >= 3:
  temp = sys.argv[2:]
  #check if provided features exist
  result = all(feature in features for feature in temp)
  if not result:
    sys.exit("Error: incorrect feature(s) supplied. Default available features are {}".format(features))
  else :
    features = temp

# test and train split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataframe[features], dataframe['increase'],test_size=.7)

model = LogisticRegression(solver = 'lbfgs')#, multi_class='multinomial'
model.fit(X_train, Y_train)

pickle.dump(model, open('model.sav', 'wb'))
