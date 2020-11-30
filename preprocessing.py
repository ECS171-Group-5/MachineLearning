import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn import preprocessing
import seaborn as sns
import sys
import pickle

if len(sys.argv) < 2:
	sys.exit("Error: incorrect number of arguments supplied. Usage: python preprocessing.py example.csv")

filepath = sys.argv[1]

# load data from csv
try:
	df = pd.read_csv(filepath)
except:
	sys.exit("Error: csv file not found.")

# drop columns that are missing more than 50% of data
valid_columns = (df.isna().sum()/df.shape[0] < .5).values
df = df.loc[:,valid_columns]

# drop remaining entries with missing values
df = df.dropna()

def add_increase_pct(grp):
   grp['increasePercent'] = grp['stockPrice'].pct_change()
   return grp

df = df.groupby('symbol').apply(add_increase_pct)

# drop symbol
df = df.drop('symbol',axis=1)

# encode day month and year as integers
df['date'] = pd.to_datetime(df['date'])
df['day'] =  df['date'].dt.day
df['month'] =  df['date'].dt.month
df['year'] =  df['date'].dt.year
df = df.drop('date',axis=1)

features = df.drop('increasePercent',axis=1).copy()

# normalize features
min_max_scaler = preprocessing.MinMaxScaler()
features = pd.DataFrame(min_max_scaler.fit_transform(features.values),columns=features.columns)

# drop features with high VIF
def get_max_vif():
	max_vif = 5
	max_feature = None
	for i in range(features.shape[1]):
		vif = variance_inflation_factor(features.values, i) 
		if vif > max_vif:
			max_vif = vif
			max_feature = features.columns[i] 
	return max_feature, max_vif

f, v = get_max_vif()
while f is not None:
	features = features.drop(f, axis=1)
	f, v = get_max_vif()


features['increasePercent'] = df['increasePercent']
df = features.copy()

# separate training and test data
test = df[df['increasePercent'].isnull()].copy().drop('increasePercent',axis=1)
train = df.drop(test.index).copy()

# encode increase percent
train['increase'] = (train['increasePercent'] > .1).astype('int64')
train['decrease'] = (train['increasePercent'] < -.1).astype('int64')
train['none'] = (train['increasePercent'].between(-.1,.1)).astype('int64')
train = train.drop('increasePercent',axis=1)

# save output
test.to_csv("test.csv",index=False)
train.to_csv("train.csv",index=False)
pickle.dump(min_max_scaler, open('scaler.sav', 'wb'))





