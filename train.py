from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import sys

if len(sys.argv) != 2:
	sys.exit("Error: incorrect number of arguments supplied. Usage: python train.py example.csv")

filepath = sys.argv[1]
names = ['book_value', 'market_cap', 'dividend_yield', 'eps', 'price_earnings_ratio', 'price_book_ratio', 'dps', 'current_ratio', 'quick_ratio', 'increase']

try:
    dataframe = pd.read_csv(filepath, names=names)
except:
    sys.exit("Error: csv file not found.")

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dataframe.drop('increase', axis=1), dataframe['increase'], test_size=.7)

model = LogisticRegression()
model.fit(X_train, Y_train)

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))