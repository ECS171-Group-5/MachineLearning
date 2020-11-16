from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import sys

if len(sys.argv) != 2:
	sys.exit("Error: incorrect number of arguments supplied. Usage: python train.py example.csv ['list','of','features','optional']")

#default features
#NOTE: removed 'increase', if allowing user to enter features we don't want them to add the target feature
names = ['book_value', 'market_cap', 'dividend_yield', 'eps', 'price_earnings_ratio', 'price_book_ratio','dps','current_ratio','quick_ratio']

#checking for input of desired features
if len(sys.argv) == 3:
  temp = sys.argv[2]
  temp = [x.lower() for x in temp]
  #check if provided features exist
  result = all(features in names for features in temp)
  if not result:
    sys.exit("Error: incorrect feature/s supplied. Default available features are ['book_value', 'market_cap', 'dividend_yield', 'eps', 'price_earnings_ratio', 'price_book_ratio','dps','current_ratio','quick_ratio']")
  else :
    names = temp

#need to create dataframe with the provided features
try:
  dataframe = pd.read_csv(filepath, header=0) #names=names
except:
  sys.exit("Error: csv file not found.")

target = dataframe['increase'].copy() # for now 'increase' is our default target feature

features = dataframe[dataframe.columns[names]].copy()

#split DF and train
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, target,test_size=.7)

model = LogisticRegression(solver = 'lbfgs')#, multi_class='multinomial'
model.fit(X_train, Y_train)

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))
