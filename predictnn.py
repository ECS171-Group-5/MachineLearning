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

df = pd.read_csv('test.csv')
scaler = pickle.load(open('scaler.sav', 'rb'))
X = scaler.transform(df.drop(['symbol'],axis=1))[['marketCap','currentRatio','quickRatio','stockPrice','month','year']]

result = model.predict_classes(X)
print(result)
