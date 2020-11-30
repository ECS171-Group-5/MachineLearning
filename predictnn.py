import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import os
import pickle

os.environ['KMP_DUPLICATE_LIB_OK']='True'

try:
	model = keras.models.load_model("nnmodel")
except:
	sys.exit("Error: model folder not found. Try training with catnn.py first.")

try:
	df = pd.read_csv('test.csv')
	scaler = pickle.load(open('scaler.sav', 'rb'))
except:
	sys.exit("Error: dataset not found. Try processing data with preprocessing.py first.")

out = pd.DataFrame()
out['symbol'] = df['symbol']

df = pd.read_csv('test.csv').drop(['symbol'],axis=1)
scaler = pickle.load(open('scaler.sav', 'rb'))
X = pd.DataFrame(scaler.transform(df),columns=df.columns)
out['predict'] = 'increasePercent'
out['result'] = model.predict_classes(X[['marketCap','currentRatio','quickRatio','stockPrice','month','year']])
print(out)
print(out.describe())
out.to_csv("output.csv",index=False)