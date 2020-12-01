import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,metrics
from imblearn.over_sampling import RandomOverSampler
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
targets = ['increase','decrease','none']
features = features.drop(targets)

# test and train split
oversample = RandomOverSampler(sampling_strategy='not majority')
X_over, y_over = oversample.fit_resample(dataframe[features].to_numpy(), dataframe[targets].to_numpy())

# Build the model
model = keras.Sequential(
    [
        layers.Dense(6,input_dim=6, activation='relu'),
        layers.Dense(6, activation='relu'),
		    layers.Dense(3, activation='softmax')
    ]
)

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=[metrics.CategoricalAccuracy(), metrics.Precision(.5),metrics.Recall(.5)]
)

# Train the model
model.fit(X_over, y_over,epochs=200, batch_size=50,class_weight={0:.35,1:.325,2:.325})

# Test the model
train_score = model.evaluate(dataframe[features],dataframe[targets],verbose=0)

print(model.metrics_names)
print(train_score)

model.save('nnmodel')
