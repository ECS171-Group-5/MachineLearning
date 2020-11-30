import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,metrics
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
# x_train, x_test, y_train, y_test = train_test_split(dataframe[features], dataframe[targets],test_size=.3)

# Build the model
model = keras.Sequential(
    [
        layers.Dense(9,input_dim=6, activation='relu'),
        layers.Dense(9, activation='elu'),
		    layers.Dense(3, activation='softmax')
    ]
)

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=[metrics.BinaryAccuracy(threshold=.5), metrics.Precision(.5),metrics.Recall(.5)]
)

# Train the model
model.fit(dataframe[features], dataframe[targets],epochs=33, batch_size=10,class_weight='balanced')

# Test the model
train_score = model.evaluate(dataframe[features],dataframe[targets],verbose=0)
# test_score = model.evaluate(x_test, y_test,verbose=0)

print(model.metrics_names)
print(train_score)

model.save('nnmodel')
