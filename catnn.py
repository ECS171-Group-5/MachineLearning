import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,metrics
import sys
import warnings
warnings.filterwarnings('ignore')

if len(sys.argv) < 2 or len(sys.argv) > 3:
	sys.exit("Error: incorrect number of arguments supplied. Usage: python train.py example.csv optional_target")

filepath = sys.argv[1]

# load data from csv
try:
  dataframe = pd.read_csv(filepath)
except:
  sys.exit("Error: csv file not found.")

features = dataframe.columns
targets = ['increase','decrease']
features = features.drop(targets)

# test and train split
x_train, x_test, y_train, y_test = train_test_split(dataframe[features], dataframe[targets],test_size=.3)

# Build the model
model = keras.Sequential()
model.add(layers.Dense(9,input_dim=9, activation='relu'))
model.add(layers.Dense(9, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy', metrics.Precision(.38),metrics.Recall(.38)]
)

# Train the model
model.fit(x_train, y_train,epochs=100, batch_size=50)

# Test the model
train_score = model.evaluate(x_train,y_train,verbose=0)
test_score = model.evaluate(x_test, y_test,verbose=0)

print(model.metrics_names)
print(test_score)